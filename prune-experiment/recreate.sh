#!/usr/bin/env bash
#
# Pruning experiment: verify |N(x)[c] - P_p(N)(x)[c]| <= epsilon for the original
# ReluDiff MNIST networks N and their globally magnitude-pruned counterparts
# P_p(N), across four verifiers:
#
#   milp_abcrown : Relational MILP (CPLEX) with alpha-beta-CROWN bound tightening
#   abcrown      : pure alpha-beta-CROWN
#   reludiff     : ReluDiff   (delta_network_test, pure ReluDiff build)
#   neurodiff    : NeuroDiff  (delta_network_test, full NeuroDiff build)
#
# The swept variable is the PRUNING RATE (percent of smallest-magnitude weights
# zeroed, global unstructured). Everything else (networks, images, input regions,
# epsilon, property) is held fixed so the only thing changing between N and P_p(N)
# is how much of N has been pruned.
#
# Usage:
#   ./recreate.sh --light     # quick smoke run (arch 3x100, prune 10% & 20%, 5 images)
#   ./recreate.sh             # full experiment (all 3 archs, prune 5..50%, 100 images)
#
# Results (clean CSVs) land in prune-experiment/results/, per-run logs in
# prune-experiment/logs/. A status summary is printed at the end.
set -uo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"

PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

DIFF_DIR="$REPO_ROOT/third_party/NeuroDiff-ASE2020-Artifact/DiffNN-Code"
RELUDIFF_BIN="$DIFF_DIR/reludiff"
NEURODIFF_BIN="$DIFF_DIR/neurodiff"

# Fixed across the sweep (suite defaults, matching the existing ReluDiff configs).
PERTURB="3"          # global perturbation strength (-p 3, i.e. +/- 3/255 per pixel)
EPSILON="1.0"        # output-difference bound
CROWN_PROFILE="relu-kfsb"

LIGHT=0
SKIP_EXISTING=0
for arg in "$@"; do
	case "$arg" in
		--light) LIGHT=1 ;;
		--skip-existing) SKIP_EXISTING=1 ;;   # resume: skip configs whose CSV exists
		-h|--help) sed -n '2,26p' "$0"; exit 0 ;;
		*) echo "unknown argument: $arg" >&2; exit 2 ;;
	esac
done

# Space-separated "arch:mode" pairs to skip entirely, e.g.
#   EXP_EXCLUDE="mnist_relu_2_512:global mnist_relu_4_1024:global"
EXCLUDE="${EXP_EXCLUDE:-}"

# Persistent skip list: glob patterns matched against each config's
# "<method>__<arch>__<mode>__p<rate>" tag. Default prune-experiment/skip.conf;
# override with EXP_SKIP_FILE=... (empty to disable).
SKIP_FILE="${EXP_SKIP_FILE-$SCRIPT_DIR/skip.conf}"
SKIP_PATTERNS=()
[ -n "$SKIP_FILE" ] && [ -f "$SKIP_FILE" ] && \
	mapfile -t SKIP_PATTERNS < <(sed 's/#.*//' "$SKIP_FILE" | awk 'NF{$1=$1;print}')

if [ "$LIGHT" -eq 1 ]; then
	ARCHS=(mnist_relu_3_100)
	RATES=(10 20)                  # pruning percentages
	MODES=(global three_pixel)     # input regions
	LIMIT=5                        # images per config
	TIMEOUT=30                     # per-instance timeout (seconds)
	MODE_LABEL="LIGHT"
else
	ARCHS=(mnist_relu_3_100 mnist_relu_2_512 mnist_relu_4_1024)
	RATES=(5 10 20 30 40 50)
	MODES=(global three_pixel)
	LIMIT=100
	TIMEOUT=60
	MODE_LABEL="FULL"
fi

METHODS=(milp_abcrown abcrown reludiff neurodiff)

# Optional environment overrides (space-separated for the list ones), so the
# full experiment can be staged without editing this file. Examples:
#   EXP_ARCHS=mnist_relu_4_1024 EXP_TIMEOUT=120 ./recreate.sh
#   EXP_METHODS="reludiff neurodiff" EXP_RATES="30 40 50" ./recreate.sh
#   EXP_MODES=global ./recreate.sh
[ -n "${EXP_ARCHS:-}" ]   && read -ra ARCHS   <<<"$EXP_ARCHS"
[ -n "${EXP_RATES:-}" ]   && read -ra RATES   <<<"$EXP_RATES"
[ -n "${EXP_MODES:-}" ]   && read -ra MODES   <<<"$EXP_MODES"
[ -n "${EXP_METHODS:-}" ] && read -ra METHODS <<<"$EXP_METHODS"
[ -n "${EXP_LIMIT:-}" ]   && LIMIT="$EXP_LIMIT"
[ -n "${EXP_TIMEOUT:-}" ] && TIMEOUT="$EXP_TIMEOUT"

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
cd "$REPO_ROOT"

SKIP_DIFF=0
if [ ! -x "$RELUDIFF_BIN" ] || [ ! -x "$NEURODIFF_BIN" ]; then
	echo "WARNING: ReluDiff/NeuroDiff binaries not found under $DIFF_DIR" >&2
	echo "         build them with: scripts/build_diffverifier.sh" >&2
	echo "         (reludiff/neurodiff methods will be skipped)" >&2
	SKIP_DIFF=1
fi

echo "=============================================================="
echo " Pruning experiment [$MODE_LABEL]"
echo "   archs   : ${ARCHS[*]}"
echo "   rates   : ${RATES[*]} (%)"
echo "   modes   : ${MODES[*]}"
echo "   images  : $LIMIT   timeout: ${TIMEOUT}s   epsilon: $EPSILON"
echo "   methods : ${METHODS[*]}"
[ -n "$EXCLUDE" ]                  && echo "   exclude : $EXCLUDE"
[ "${#SKIP_PATTERNS[@]}" -gt 0 ]   && echo "   skipcfg : ${#SKIP_PATTERNS[@]} pattern(s) from ${SKIP_FILE#$REPO_ROOT/}"
[ "$SKIP_EXISTING" -eq 1 ]         && echo "   resume  : skipping configs whose CSV already exists"
echo "   results : $RESULTS_DIR"
echo "=============================================================="

# --------------------------------------------------------------------------- #
# Run one (method, arch, rate) config, writing a clean CSV via --csv.
# --------------------------------------------------------------------------- #
declare -a SUMMARY

sparsity_of() { awk "BEGIN{printf \"%.4f\", $1/100}"; }

common_opts() {
	local arch="$1" sparsity="$2" mode="$3"
	printf '%s\n' \
		--suite mnist_reludiff \
		--suite-options "networks=$arch" \
		--suite-options "modes=$mode" \
		--suite-options "perturbation=prune" \
		--suite-options "sparsity=$sparsity" \
		--suite-options "perturb=$PERTURB" \
		--suite-options "epsilon=$EPSILON" \
		--suite-options "limit=$LIMIT" \
		--suite-options "timeout=$TIMEOUT"
}

run_config() {
	local method="$1" arch="$2" rate="$3" mode="$4"
	local sparsity out log tag
	sparsity="$(sparsity_of "$rate")"
	tag="${method}__${arch}__${mode}__p${rate}"
	out="$RESULTS_DIR/${tag}.csv"
	log="$LOGS_DIR/${tag}.log"

	# Persistent skip list (skip.conf): glob match against the full tag.
	local pat
	if [ "${#SKIP_PATTERNS[@]}" -gt 0 ]; then
		for pat in "${SKIP_PATTERNS[@]}"; do
			if [[ $tag == $pat ]]; then
				printf -- '--- %-13s arch=%-18s mode=%-11s prune=%s%% skip (skip.conf)\n' \
					"$method" "$arch" "$mode" "$rate"
				SUMMARY+=("$method|$arch|$mode|$rate|skipconf")
				return
			fi
		done
	fi

	# Exclude specific arch:mode pairs (EXP_EXCLUDE).
	local ex
	for ex in $EXCLUDE; do
		if [ "$arch:$mode" = "$ex" ]; then
			printf -- '--- %-13s arch=%-18s mode=%-11s prune=%s%% EXCLUDED\n' \
				"$method" "$arch" "$mode" "$rate"
			SUMMARY+=("$method|$arch|$mode|$rate|excluded")
			return
		fi
	done

	# Resume: skip configs already gathered (CSV present with data rows).
	if [ "$SKIP_EXISTING" -eq 1 ] && [ "$(wc -l <"$out" 2>/dev/null || echo 0)" -gt 1 ]; then
		printf -- '--- %-13s arch=%-18s mode=%-11s prune=%s%% skip (exists)\n' \
			"$method" "$arch" "$mode" "$rate"
		SUMMARY+=("$method|$arch|$mode|$rate|exists")
		return
	fi

	local -a opts
	mapfile -t opts < <(common_opts "$arch" "$sparsity" "$mode")

	local -a cmd
	case "$method" in
		milp_abcrown)
			cmd=("$PY" -m benchmarks.run_pyomo "${opts[@]}"
			     --solver cplex --bound-tightening abcrown --csv "$out") ;;
		abcrown)
			cmd=("$PY" -m benchmarks.run_crown "${opts[@]}"
			     --profile "$CROWN_PROFILE" --csv "$out") ;;
		reludiff)
			[ "$SKIP_DIFF" -eq 1 ] && { SUMMARY+=("$method|$arch|$mode|$rate|skipped"); return; }
			cmd=("$PY" -m benchmarks.run_diffverifier --tool reludiff
			     --binary "$RELUDIFF_BIN" "${opts[@]}" --csv "$out") ;;
		neurodiff)
			[ "$SKIP_DIFF" -eq 1 ] && { SUMMARY+=("$method|$arch|$mode|$rate|skipped"); return; }
			cmd=("$PY" -m benchmarks.run_diffverifier --tool neurodiff
			     --binary "$NEURODIFF_BIN" "${opts[@]}" --csv "$out") ;;
	esac

	printf '>>> %-13s arch=%-18s mode=%-11s prune=%s%%\n' "$method" "$arch" "$mode" "$rate"
	local t0 t1 rc
	t0=$(date +%s)
	"${cmd[@]}" >"$log" 2>&1
	rc=$?
	t1=$(date +%s)
	local elapsed=$((t1 - t0))
	if [ $rc -eq 0 ]; then
		printf '    done in %ss -> %s\n' "$elapsed" "${out#$REPO_ROOT/}"
		SUMMARY+=("$method|$arch|$mode|$rate|ok(${elapsed}s)")
	else
		printf '    FAILED (rc=%s) in %ss; see %s\n' "$rc" "$elapsed" "${log#$REPO_ROOT/}"
		SUMMARY+=("$method|$arch|$mode|$rate|FAIL(rc=$rc)")
	fi
}

for method in "${METHODS[@]}"; do
	for arch in "${ARCHS[@]}"; do
		for mode in "${MODES[@]}"; do
			for rate in "${RATES[@]}"; do
				run_config "$method" "$arch" "$rate" "$mode"
			done
		done
	done
done

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo
echo "=============================== summary ==============================="
printf '%-14s %-18s %-12s %-7s %s\n' "method" "arch" "mode" "prune%" "outcome"
for row in "${SUMMARY[@]}"; do
	IFS='|' read -r m a md r o <<<"$row"
	printf '%-14s %-18s %-12s %-7s %s\n' "$m" "$a" "$md" "$r" "$o"
done
echo "======================================================================"
echo "per-config status/runtime rollups:"
for csv in "$RESULTS_DIR"/*.csv; do
	[ -e "$csv" ] || continue
	echo "--- ${csv#$REPO_ROOT/}"
	"$PY" "$REPO_ROOT/summarize_out_csv.py" "$csv" 2>/dev/null \
		| grep -E "row_count|status_|sum_runtime" | sed 's/^/    /'
done
