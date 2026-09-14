#!/usr/bin/env bash
#
# Distillation experiment: verify logit equivalence |z_T(x)[c] - z_S(x)[c]| <=
# epsilon between a knowledge-distilled TEACHER (z_T) and STUDENT (z_S) on the
# labeled class c, over the ReluDiff 100-image / 3-pixel MNIST fixtures.
#
# Two experimental tiers (defined in training/distill_mnist.py):
#
#   Tier A  same architecture   kd_a1 (784-64-32-10), kd_a2 (784-128-64-10)
#           -> fair head-to-head across ALL FOUR verifiers.
#   Tier B  different architecture   kd_1, kd_2, kd_3
#           -> ReluDiff/NeuroDiff require identical archs and CANNOT run these,
#              so Tier B is Relational-MILP / ab-CROWN only.
#
# Verifiers:
#   milp_abcrown : Relational MILP (CPLEX) with alpha-beta-CROWN bound tightening
#   abcrown      : pure alpha-beta-CROWN
#   reludiff     : ReluDiff   (delta_network_test, pure ReluDiff build)   [Tier A]
#   neurodiff    : NeuroDiff  (delta_network_test, full NeuroDiff build)  [Tier A]
#
# Usage:
#   ./recreate.sh --light        # smoke run (kd_a1 + kd_1, three_pixel, 5 images)
#   ./recreate.sh                # full experiment (all 5 pairs, 100 images)
#   ./recreate.sh --train        # (re)train every pair first, then run
#   ./recreate.sh --train-only   # only (re)train the pairs, run nothing
#
# The epsilon is fixed across the sweep (EXP_EPSILON, default 2.0) so every
# verifier checks the same property per pair; set EXP_EPSILON=auto to use each
# pair's p95 logit-gap from its metadata.json instead. Each pair's logit-gap
# stats are printed at preflight so you can pick a defensible value.
#
# Results (clean CSVs) land in distillation/results/, per-run logs in
# distillation/logs/. A status summary is printed at the end.
set -uo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$REPO_ROOT/data/distillation/mnist"

PY="${EXP_PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

DIFF_DIR="$REPO_ROOT/third_party/NeuroDiff-ASE2020-Artifact/DiffNN-Code"
RELUDIFF_BIN="$DIFF_DIR/reludiff"
NEURODIFF_BIN="$DIFF_DIR/neurodiff"

CROWN_PROFILE="relu-kfsb"

LIGHT=0
TRAIN=0
TRAIN_ONLY=0
SKIP_EXISTING=0
for arg in "$@"; do
	case "$arg" in
		--light) LIGHT=1 ;;
		--train) TRAIN=1 ;;                   # (re)train pairs, then run
		--train-only) TRAIN=1; TRAIN_ONLY=1 ;; # only (re)train, run nothing
		--skip-existing) SKIP_EXISTING=1 ;;   # resume: skip configs whose CSV exists
		-h|--help) sed -n '2,33p' "$0"; exit 0 ;;
		*) echo "unknown argument: $arg" >&2; exit 2 ;;
	esac
done

# Tier membership. Tier A pairs run on all four verifiers; Tier B pairs only on
# the architecture-flexible ones (milp_abcrown, abcrown).
TIER_A_PAIRS=(kd_a1 kd_a2)
TIER_B_PAIRS=(kd_1 kd_2 kd_3)

if [ "$LIGHT" -eq 1 ]; then
	TIER_A_PAIRS=(kd_a1)
	TIER_B_PAIRS=(kd_1)
	MODES=(three_pixel)
	LIMIT=5
	TIMEOUT=30
	MODE_LABEL="LIGHT"
else
	MODES=(three_pixel)
	LIMIT=100
	TIMEOUT=30
	MODE_LABEL="FULL"
fi

METHODS=(milp_abcrown abcrown reludiff neurodiff)
EPSILON="${EXP_EPSILON:-2.0}"   # fixed per-pair bound, or "auto" for p95 logit gap
PERTURB="3"                     # global-mode perturbation (only used if modes=global)

# Optional environment overrides (space-separated for the list ones):
#   EXP_TIER_A="kd_a1" EXP_TIMEOUT=60 ./recreate.sh
#   EXP_METHODS="milp_abcrown abcrown" EXP_MODES=global ./recreate.sh
#   EXP_EPSILON=auto ./recreate.sh
[ -n "${EXP_TIER_A:-}" ]  && read -ra TIER_A_PAIRS <<<"$EXP_TIER_A"
[ -n "${EXP_TIER_B:-}" ]  && read -ra TIER_B_PAIRS <<<"$EXP_TIER_B"
[ -n "${EXP_MODES:-}" ]   && read -ra MODES   <<<"$EXP_MODES"
[ -n "${EXP_METHODS:-}" ] && read -ra METHODS <<<"$EXP_METHODS"
[ -n "${EXP_LIMIT:-}" ]   && LIMIT="$EXP_LIMIT"
[ -n "${EXP_TIMEOUT:-}" ] && TIMEOUT="$EXP_TIMEOUT"

ALL_PAIRS=("${TIER_A_PAIRS[@]}" "${TIER_B_PAIRS[@]}")

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
cd "$REPO_ROOT"

# tier_of PAIR -> "A" or "B"
tier_of() {
	local p="$1" a
	for a in "${TIER_A_PAIRS[@]}"; do [ "$a" = "$p" ] && { echo A; return; }; done
	echo B
}

# (Re)train the pairs via training/distill_mnist.py (--force overwrites).
train_pairs() {
	echo "=============================================================="
	echo " Training distillation pairs: ${ALL_PAIRS[*]}"
	echo "=============================================================="
	local p rc
	for p in "${ALL_PAIRS[@]}"; do
		echo ">>> train $p"
		"$PY" -m training.distill_mnist --pair-id "$p" --force \
			>"$LOGS_DIR/train-${p}.log" 2>&1
		rc=$?
		if [ $rc -eq 0 ]; then
			echo "    done -> ${DATA_DIR#$REPO_ROOT/}/$p"
		else
			echo "    FAILED (rc=$rc); see ${LOGS_DIR#$REPO_ROOT/}/train-${p}.log" >&2
		fi
	done
}

if [ "$TRAIN" -eq 1 ]; then
	train_pairs
fi
if [ "$TRAIN_ONLY" -eq 1 ]; then
	exit 0
fi

# Verify every requested pair has trained data on disk.
missing=()
for p in "${ALL_PAIRS[@]}"; do
	[ -f "$DATA_DIR/$p/metadata.json" ] || missing+=("$p")
done
if [ "${#missing[@]}" -gt 0 ]; then
	echo "ERROR: missing trained pair(s): ${missing[*]}" >&2
	echo "       train them first: ./recreate.sh --train  (or --train-only)" >&2
	exit 1
fi

SKIP_DIFF=0
if [ ! -x "$RELUDIFF_BIN" ] || [ ! -x "$NEURODIFF_BIN" ]; then
	echo "WARNING: ReluDiff/NeuroDiff binaries not found under $DIFF_DIR" >&2
	echo "         build them with: scripts/build_diffverifier.sh" >&2
	echo "         (reludiff/neurodiff methods will be skipped; Tier A loses its" >&2
	echo "          cross-family comparison)" >&2
	SKIP_DIFF=1
fi

# epsilon_for PAIR -> fixed EPSILON, or the pair's p95 logit gap when EXP_EPSILON=auto.
epsilon_for() {
	local pair="$1"
	if [ "$EPSILON" != "auto" ]; then
		echo "$EPSILON"
		return
	fi
	"$PY" - "$DATA_DIR/$pair/metadata.json" <<-'PYEOF'
	import json, sys
	meta = json.load(open(sys.argv[1]))
	gap = meta.get("logit_gap_on_reludiff_centers", {})
	print(f"{gap.get('p95', 1.0):.4f}")
	PYEOF
}

echo "=============================================================="
echo " Distillation experiment [$MODE_LABEL]"
echo "   tier A  : ${TIER_A_PAIRS[*]}   (all verifiers)"
echo "   tier B  : ${TIER_B_PAIRS[*]}   (milp_abcrown, abcrown only)"
echo "   modes   : ${MODES[*]}"
echo "   images  : $LIMIT   timeout: ${TIMEOUT}s   epsilon: $EPSILON"
echo "   methods : ${METHODS[*]}"
[ "$SKIP_EXISTING" -eq 1 ] && echo "   resume  : skipping configs whose CSV already exists"
echo "   results : $RESULTS_DIR"
echo "--------------------------------------------------------------"
echo " logit-gap stats (from each pair's metadata.json):"
for p in "${ALL_PAIRS[@]}"; do
	"$PY" - "$p" "$DATA_DIR/$p/metadata.json" <<-'PYEOF'
	import json, sys
	pair, path = sys.argv[1], sys.argv[2]
	g = json.load(open(path)).get("logit_gap_on_reludiff_centers", {})
	print(f"   {pair:<7} median={g.get('median', float('nan')):.3f} "
	      f"p95={g.get('p95', float('nan')):.3f} max={g.get('max', float('nan')):.3f}")
	PYEOF
done
echo "=============================================================="

# --------------------------------------------------------------------------- #
# Build one verifier invocation into the global CMD array.
# Returns 1 when a required ReluDiff/NeuroDiff binary is missing.
# --------------------------------------------------------------------------- #
declare -a CMD
build_cmd() {
	local method="$1" pair="$2" mode="$3" eps="$4" out="$5"
	local -a opts=(
		--suite distillation
		--suite-options "pairs=$pair"
		--suite-options "modes=$mode"
		--suite-options "epsilon=$eps"
		--suite-options "perturb=$PERTURB"
		--suite-options "limit=$LIMIT"
		--suite-options "timeout=$TIMEOUT"
	)
	case "$method" in
		milp_abcrown)
			CMD=("$PY" -m benchmarks.run_pyomo "${opts[@]}"
			     --solver cplex --bound-tightening abcrown --csv "$out") ;;
		abcrown)
			CMD=("$PY" -m benchmarks.run_crown "${opts[@]}"
			     --profile "$CROWN_PROFILE" --csv "$out") ;;
		reludiff)
			[ "$SKIP_DIFF" -eq 1 ] && return 1
			CMD=("$PY" -m benchmarks.run_diffverifier --tool reludiff
			     --binary "$RELUDIFF_BIN" "${opts[@]}" --csv "$out") ;;
		neurodiff)
			[ "$SKIP_DIFF" -eq 1 ] && return 1
			CMD=("$PY" -m benchmarks.run_diffverifier --tool neurodiff
			     --binary "$NEURODIFF_BIN" "${opts[@]}" --csv "$out") ;;
		*) echo "unknown method: $method" >&2; return 2 ;;
	esac
	return 0
}

declare -a SUMMARY

run_config() {
	local method="$1" pair="$2" mode="$3"
	local tier out log tag eps
	tier="$(tier_of "$pair")"

	# ReluDiff/NeuroDiff cannot run diff-arch (Tier B) pairs.
	if [ "$tier" = "B" ] && { [ "$method" = "reludiff" ] || [ "$method" = "neurodiff" ]; }; then
		printf -- '--- %-13s pair=%-6s mode=%-11s N/A (Tier B, arch differs)\n' \
			"$method" "$pair" "$mode"
		SUMMARY+=("$method|$pair|$tier|$mode|n/a-tierB")
		return
	fi

	eps="$(epsilon_for "$pair")"
	tag="${method}__${pair}__${mode}"
	out="$RESULTS_DIR/${tag}.csv"
	log="$LOGS_DIR/${tag}.log"

	# Resume: skip configs already gathered (CSV present with data rows).
	if [ "$SKIP_EXISTING" -eq 1 ] && [ "$(wc -l <"$out" 2>/dev/null || echo 0)" -gt 1 ]; then
		printf -- '--- %-13s pair=%-6s mode=%-11s skip (exists)\n' "$method" "$pair" "$mode"
		SUMMARY+=("$method|$pair|$tier|$mode|exists")
		return
	fi

	if ! build_cmd "$method" "$pair" "$mode" "$eps" "$out"; then
		printf -- '--- %-13s pair=%-6s mode=%-11s skipped (no binary)\n' "$method" "$pair" "$mode"
		SUMMARY+=("$method|$pair|$tier|$mode|skipped")
		return
	fi

	printf '>>> %-13s pair=%-6s (tier %s) mode=%-11s eps=%s\n' \
		"$method" "$pair" "$tier" "$mode" "$eps"
	local t0 t1 rc elapsed
	t0=$(date +%s)
	"${CMD[@]}" >"$log" 2>&1
	rc=$?
	t1=$(date +%s)
	elapsed=$((t1 - t0))
	if [ $rc -eq 0 ]; then
		printf '    done in %ss -> %s\n' "$elapsed" "${out#$REPO_ROOT/}"
		SUMMARY+=("$method|$pair|$tier|$mode|ok(${elapsed}s)")
	else
		printf '    FAILED (rc=%s) in %ss; see %s\n' "$rc" "$elapsed" "${log#$REPO_ROOT/}"
		SUMMARY+=("$method|$pair|$tier|$mode|FAIL(rc=$rc)")
	fi
}

for method in "${METHODS[@]}"; do
	for pair in "${ALL_PAIRS[@]}"; do
		for mode in "${MODES[@]}"; do
			run_config "$method" "$pair" "$mode"
		done
	done
done

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo
echo "=============================== summary ==============================="
printf '%-14s %-7s %-5s %-12s %s\n' "method" "pair" "tier" "mode" "outcome"
for row in "${SUMMARY[@]}"; do
	IFS='|' read -r m p t md o <<<"$row"
	printf '%-14s %-7s %-5s %-12s %s\n' "$m" "$p" "$t" "$md" "$o"
done
echo "======================================================================"
echo "per-config status/runtime rollups:"
for csv in "$RESULTS_DIR"/*.csv; do
	[ -e "$csv" ] || continue
	echo "--- ${csv#$REPO_ROOT/}"
	"$PY" "$REPO_ROOT/summarize_out_csv.py" "$csv" 2>/dev/null \
		| grep -E "row_count|status_|sum_runtime" | sed 's/^/    /'
done
