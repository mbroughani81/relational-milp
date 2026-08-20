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
#   ./recreate.sh --audit-skips [--audit-limit=N]
#                             # audit skip.conf instead of running the experiment
#
# Results (clean CSVs) land in prune-experiment/results/, per-run logs in
# prune-experiment/logs/. A status summary is printed at the end.
#
# --audit-skips exists because skip.conf entries are only legitimate when the
# config really is hopeless. It expands every skip.conf pattern over the config
# grid and reruns each matching config on the first N instances (default 10, same
# timeout as the real sweep). A config that solves >= 1 instance is reported as
# "REMOVE from skip.conf"; a config where all N instances time out is appended to
# prune-experiment/skip-confirmed.conf and is not probed again on later audits.
# Audit CSVs go to prune-experiment/audit/ and never overwrite results/.
set -uo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"
AUDIT_DIR="$SCRIPT_DIR/audit"

PY="${EXP_PY:-$REPO_ROOT/.venv/bin/python}"
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
AUDIT=0
AUDIT_LIMIT=10
for arg in "$@"; do
	case "$arg" in
		--light) LIGHT=1 ;;
		--skip-existing) SKIP_EXISTING=1 ;;   # resume: skip configs whose CSV exists
		--audit-skips) AUDIT=1 ;;             # probe skip.conf entries, run nothing else
		--audit-limit=*) AUDIT_LIMIT="${arg#*=}" ;;
		-h|--help) sed -n '2,32p' "$0"; exit 0 ;;
		*) echo "unknown argument: $arg" >&2; exit 2 ;;
	esac
done
case "$AUDIT_LIMIT" in
	''|*[!0-9]*|0) echo "--audit-limit must be a positive integer" >&2; exit 2 ;;
esac

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

# Configs --audit-skips has already proven hopeless (exact tags, one per line).
# They are not re-probed. Set EXP_AUDIT_CONFIRMED_FILE= (empty) to re-probe all.
CONFIRMED_FILE="${EXP_AUDIT_CONFIRMED_FILE-$SCRIPT_DIR/skip-confirmed.conf}"

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
	local arch="$1" sparsity="$2" mode="$3" limit="$4"
	printf '%s\n' \
		--suite mnist_reludiff \
		--suite-options "networks=$arch" \
		--suite-options "modes=$mode" \
		--suite-options "perturbation=prune" \
		--suite-options "sparsity=$sparsity" \
		--suite-options "perturb=$PERTURB" \
		--suite-options "epsilon=$EPSILON" \
		--suite-options "limit=$limit" \
		--suite-options "timeout=$TIMEOUT"
}

# Fill the global CMD array with the verifier invocation for one config.
# Returns 1 when the method needs a ReluDiff/NeuroDiff binary that is missing.
declare -a CMD
build_cmd() {
	local method="$1" arch="$2" mode="$3" rate="$4" limit="$5" out="$6"
	local -a opts
	mapfile -t opts < <(common_opts "$arch" "$(sparsity_of "$rate")" "$mode" "$limit")
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

# Print every skip.conf pattern that matches a config tag.
matching_skip_patterns() {
	local tag="$1" pat
	for pat in ${SKIP_PATTERNS[@]+"${SKIP_PATTERNS[@]}"}; do
		[[ $tag == $pat ]] && printf '%s\n' "$pat"
	done
}

# rows solved timeout other  <- status tallies of a results CSV
csv_status_counts() {
	awk -F, '
		NR == 1 { for (i = 1; i <= NF; i++) if ($i == "status") col = i; next }
		col && NF >= col {
			rows++
			s = $col
			if (s == "unsat" || s == "sat" || s == "verified" || s == "falsified" ||
			    s == "safe" || s == "verified-by-bounds" || s == "unsafe-pgd" ||
			    s == "unsafe-bab" || s == "adv found") solved++
			else if (s == "timeout") timeouts++
			else other++
		}
		END { printf "%d %d %d %d\n", rows + 0, solved + 0, timeouts + 0, other + 0 }
	' "$1" 2>/dev/null || echo "0 0 0 0"
}

run_config() {
	local method="$1" arch="$2" rate="$3" mode="$4"
	local out log tag
	tag="${method}__${arch}__${mode}__p${rate}"
	out="$RESULTS_DIR/${tag}.csv"
	log="$LOGS_DIR/${tag}.log"

	# Persistent skip list (skip.conf): glob match against the full tag.
	if [ -n "$(matching_skip_patterns "$tag")" ]; then
		printf -- '--- %-13s arch=%-18s mode=%-11s prune=%s%% skip (skip.conf)\n' \
			"$method" "$arch" "$mode" "$rate"
		SUMMARY+=("$method|$arch|$mode|$rate|skipconf")
		return
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

	if ! build_cmd "$method" "$arch" "$mode" "$rate" "$LIMIT" "$out"; then
		SUMMARY+=("$method|$arch|$mode|$rate|skipped")
		return
	fi

	printf '>>> %-13s arch=%-18s mode=%-11s prune=%s%%\n' "$method" "$arch" "$mode" "$rate"
	local t0 t1 rc
	t0=$(date +%s)
	"${CMD[@]}" >"$log" 2>&1
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

# --------------------------------------------------------------------------- #
# --audit-skips: probe every skip.conf config on a few instances and report the
# ones that are NOT hopeless, i.e. that solve at least one instance.
#
# Audit runs never touch results/: CSVs go to prune-experiment/audit/ and logs
# to logs/audit-<tag>.log. Configs proven to time out on every probed instance
# are appended to the confirmed file so later audits skip them.
# --------------------------------------------------------------------------- #
audit_skips() {
	local audited=0
	local -a to_unskip=() confirmed=() inconclusive=()

	mkdir -p "$AUDIT_DIR"
	local -a already=()
	[ -n "$CONFIRMED_FILE" ] && [ -f "$CONFIRMED_FILE" ] && \
		mapfile -t already < <(sed 's/#.*//' "$CONFIRMED_FILE" | awk 'NF{$1=$1;print}')

	echo "=============================================================="
	echo " Auditing skip.conf entries"
	echo "   patterns  : ${#SKIP_PATTERNS[@]} from ${SKIP_FILE#$REPO_ROOT/}"
	echo "   probe     : first $AUDIT_LIMIT instance(s), timeout ${TIMEOUT}s"
	echo "   confirmed : ${CONFIRMED_FILE:-<disabled>} (${#already[@]} entry/entries already proven)"
	echo "   csvs      : ${AUDIT_DIR#$REPO_ROOT/}"
	echo "=============================================================="

	local method arch mode rate tag out log known t0 t1 rc counts
	local rows solved timeouts other elapsed verdict
	for method in "${METHODS[@]}"; do
		for arch in "${ARCHS[@]}"; do
			for mode in "${MODES[@]}"; do
				for rate in "${RATES[@]}"; do
					tag="${method}__${arch}__${mode}__p${rate}"
					[ -z "$(matching_skip_patterns "$tag")" ] && continue

					known=0
					for pat in ${already[@]+"${already[@]}"}; do
						[ "$tag" = "$pat" ] && known=1 && break
					done
					if [ "$known" -eq 1 ]; then
						printf -- '--- %-58s already confirmed timeout\n' "$tag"
						continue
					fi

					out="$AUDIT_DIR/${tag}.csv"
					log="$LOGS_DIR/audit-${tag}.log"
					if ! build_cmd "$method" "$arch" "$mode" "$rate" "$AUDIT_LIMIT" "$out"; then
						printf -- '--- %-58s no binary, cannot audit\n' "$tag"
						inconclusive+=("$tag|verifier binary missing")
						continue
					fi

					printf '>>> %-58s probing %s instance(s)\n' "$tag" "$AUDIT_LIMIT"
					audited=$((audited + 1))
					t0=$(date +%s)
					"${CMD[@]}" >"$log" 2>&1
					rc=$?
					t1=$(date +%s)
					elapsed=$((t1 - t0))

					counts="$(csv_status_counts "$out")"
					read -r rows solved timeouts other <<<"$counts"
					if [ "$rows" -eq 0 ]; then
						verdict="no rows (rc=$rc, see ${log#$REPO_ROOT/})"
						inconclusive+=("$tag|$verdict")
					elif [ "$solved" -gt 0 ]; then
						verdict="SOLVES $solved/$rows -> remove from skip.conf"
						to_unskip+=("$tag|$solved/$rows solved")
					elif [ "$timeouts" -eq "$rows" ]; then
						verdict="all $rows timed out -> confirmed"
						confirmed+=("$tag|$rows/$rows timeout @${TIMEOUT}s")
					else
						verdict="no solve, but $other non-timeout row(s) -> inconclusive"
						inconclusive+=("$tag|$solved solved, $timeouts timeout, $other other of $rows")
					fi
					printf '    %ss: %s\n' "$elapsed" "$verdict"
				done
			done
		done
	done

	# Persist the proven-hopeless configs so a later audit does not rerun them.
	if [ -n "$CONFIRMED_FILE" ] && [ "${#confirmed[@]}" -gt 0 ]; then
		if [ ! -f "$CONFIRMED_FILE" ]; then
			{
				echo "# Configs proven to time out on every probed instance by"
				echo "# ./recreate.sh --audit-skips. Exact tags, one per line."
				echo "# Auditing skips anything listed here; delete a line to re-probe it."
			} >"$CONFIRMED_FILE"
		fi
		local entry
		for entry in "${confirmed[@]}"; do
			IFS='|' read -r tag note <<<"$entry"
			printf '%-58s # %s, %s instance(s), audited %s\n' \
				"$tag" "$note" "$AUDIT_LIMIT" "$(date +%Y-%m-%d)" >>"$CONFIRMED_FILE"
		done
	fi

	echo
	echo "============================ audit summary ============================"
	printf 'audited %s config(s): %s solve, %s confirmed timeout, %s inconclusive\n' \
		"$audited" "${#to_unskip[@]}" "${#confirmed[@]}" "${#inconclusive[@]}"

	echo
	echo "--- REMOVE from ${SKIP_FILE#$REPO_ROOT/} (these solve instances) ---"
	if [ "${#to_unskip[@]}" -eq 0 ]; then
		echo "    (none)"
	else
		local pat
		for entry in "${to_unskip[@]}"; do
			IFS='|' read -r tag note <<<"$entry"
			printf '    %-58s %s\n' "$tag" "$note"
			while read -r pat; do
				[ -n "$pat" ] && printf '        matched by pattern: %s\n' "$pat"
			done < <(matching_skip_patterns "$tag")
		done
		echo
		echo "    Patterns above must be deleted or narrowed so these tags run again."
	fi

	if [ "${#inconclusive[@]}" -gt 0 ]; then
		echo
		echo "--- INCONCLUSIVE (check the logs) ---"
		for entry in "${inconclusive[@]}"; do
			IFS='|' read -r tag note <<<"$entry"
			printf '    %-58s %s\n' "$tag" "$note"
		done
	fi

	if [ "${#confirmed[@]}" -gt 0 ]; then
		echo
		echo "--- CONFIRMED timeout (recorded in ${CONFIRMED_FILE#$REPO_ROOT/}) ---"
		for entry in "${confirmed[@]}"; do
			IFS='|' read -r tag note <<<"$entry"
			printf '    %-58s %s\n' "$tag" "$note"
		done
	fi
	echo "======================================================================"
}

if [ "$AUDIT" -eq 1 ]; then
	audit_skips
	exit 0
fi

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
