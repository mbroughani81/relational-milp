#!/usr/bin/env bash
#
# Verda fleet worker: pulls (method,arch,mode,rate) cells from a shared work
# queue and runs each one via prune-experiment/recreate.sh, writing all results
# to the shared filesystem. Every node in the fleet runs this IDENTICAL script;
# the queue self-balances (slow milp_abcrown cells spread across nodes, fast
# reludiff/neurodiff cells get mopped up), so scaling is just "add more nodes".
#
# Prereqs on each node (see README + setup.sh):
#   * shared volume mounted at $SHARED (default /mnt/exp-data)
#   * CPLEX subtree already copied to $SHARED/ibm/CPLEX_Studio222/cplex  (once)
#   * ./setup.sh already run, e.g.:
#       CPLEX_HOME=$SHARED/ibm/CPLEX_Studio222 ./setup.sh
#
# Usage (from repo root):
#   scripts/verda_worker.sh                 # work the queue until it is drained
#   PROGRESS=1 scripts/verda_worker.sh      # just print queue progress and exit
#
# Env overrides:
#   SHARED     shared-FS mount              (default /mnt/exp-data)
#   RUN        run name / subdir under it   (default prune-10min)
#   EXP_METHODS EXP_ARCHS EXP_MODES EXP_RATES EXP_LIMIT EXP_TIMEOUT
#              the sweep grid (defaults below match the "10-min, all archs,
#              relational-milp/reludiff/neurodiff, global mode" request)
#   RECLAIM_STALE_SEC  reclaim a claimed-but-unfinished cell whose lock is older
#              than this many seconds (default 0 = never; a milp cell can
#              legitimately run ~16h, so only enable this for known-dead nodes).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# The sweep grid. Defaults encode the request: relational-milp + the two diff
# verifiers, all three archs, global mode (the only mode valid ACROSS families
# -- three_pixel opens different pixels for MILP vs ReluDiff/NeuroDiff), 100
# images, 10-minute (600s) per-instance timeout.
# --------------------------------------------------------------------------- #
read -ra METHODS <<<"${EXP_METHODS:-milp_abcrown reludiff neurodiff}"
read -ra ARCHS   <<<"${EXP_ARCHS:-mnist_relu_3_100 mnist_relu_2_512 mnist_relu_4_1024}"
read -ra MODES   <<<"${EXP_MODES:-global}"
read -ra RATES   <<<"${EXP_RATES:-5 10 20 30 40 50}"
LIMIT="${EXP_LIMIT:-100}"
TIMEOUT="${EXP_TIMEOUT:-600}"

SHARED="${SHARED:-/mnt/exp-data}"
RUN="${RUN:-prune-10min}"
OUT="$SHARED/$RUN"
RECLAIM_STALE_SEC="${RECLAIM_STALE_SEC:-0}"

# Disable recreate.sh's skip.conf so NOTHING is skipped: every cell runs its full
# LIMIT images at the full TIMEOUT, including the ones skip.conf marks hopeless.
# (Empty EXP_SKIP_FILE turns skipping off; set SKIP_FILE=prune-experiment/skip.conf
# to restore the curated skips.) This does NOT affect --skip-existing resume.
SKIP_FILE="${SKIP_FILE:-}"

[ -d "$SHARED" ] || { echo "ERROR: shared FS not mounted at $SHARED" >&2; exit 1; }
mkdir -p "$OUT/results" "$OUT/logs" "$OUT/queue"

# --------------------------------------------------------------------------- #
# Route recreate.sh's hardcoded results/ and logs/ dirs onto the shared FS via
# symlinks. Guard the PRECIOUS local results dir: never clobber a real, non-empty
# prune-experiment/results (those CSVs are gitignored and costly). On a fresh
# Verda clone the dir is empty/absent, so this is a no-op replace.
# --------------------------------------------------------------------------- #
link_to_shared() {
	local local_dir="$1" shared_dir="$2"
	if [ -L "$local_dir" ]; then ln -sfn "$shared_dir" "$local_dir"; return; fi
	if [ -d "$local_dir" ] && [ -n "$(ls -A "$local_dir" 2>/dev/null)" ]; then
		echo "ERROR: $local_dir is a non-empty real directory; refusing to replace it" >&2
		echo "       (protecting existing result CSVs). Move it aside or run on a clean clone." >&2
		exit 1
	fi
	rm -rf "$local_dir"
	ln -sfn "$shared_dir" "$local_dir"
}
link_to_shared "$REPO_ROOT/prune-experiment/results" "$OUT/results"
link_to_shared "$REPO_ROOT/prune-experiment/logs"    "$OUT/logs"

tags() {
	local m a md r
	for m in "${METHODS[@]}"; do for a in "${ARCHS[@]}"; do
		for md in "${MODES[@]}"; do for r in "${RATES[@]}"; do
			printf '%s\n' "${m}__${a}__${md}__p${r}"
		done; done
	done; done
}

cell_done() {  # a cell is done when its results CSV has a header + >=1 data row
	[ "$(wc -l <"$OUT/results/$1.csv" 2>/dev/null || echo 0)" -gt 1 ]
}

if [ "${PROGRESS:-0}" = "1" ]; then
	total=0 done=0 claimed=0
	while read -r tag; do
		total=$((total+1))
		if cell_done "$tag"; then done=$((done+1))
		elif [ -d "$OUT/queue/$tag.lock" ]; then claimed=$((claimed+1)); fi
	done < <(tags)
	echo "run=$RUN  done=$done/$total  in-progress=$claimed  pending=$((total-done-claimed))"
	exit 0
fi

HOST="$(hostname)"
echo "worker $HOST starting on queue $OUT (methods=${METHODS[*]} archs=${ARCHS[*]} modes=${MODES[*]} rates=${RATES[*]} limit=$LIMIT timeout=${TIMEOUT}s)"

worked=0
while read -r tag; do
	cell_done "$tag" && continue
	lock="$OUT/queue/$tag.lock"

	# Atomic claim: mkdir succeeds for exactly one node on the shared FS.
	if ! mkdir "$lock" 2>/dev/null; then
		# Already claimed. Optionally reclaim if the owner looks dead.
		if [ "$RECLAIM_STALE_SEC" -gt 0 ] && ! cell_done "$tag"; then
			age=$(( $(date +%s) - $(stat -c %Y "$lock" 2>/dev/null || date +%s) ))
			if [ "$age" -gt "$RECLAIM_STALE_SEC" ]; then
				echo "reclaiming stale cell $tag (lock age ${age}s)"
				rm -rf "$lock"; mkdir "$lock" 2>/dev/null || continue
			else
				continue
			fi
		else
			continue
		fi
	fi
	printf '%s\t%s\n' "$HOST" "$(date -u +%FT%TZ)" >"$lock/owner"

	# Decompose the tag and run exactly that one cell through recreate.sh. Reuse
	# its tested command-building, skip.conf handling, and CSV writer. --skip-
	# existing double-guards against a race where the CSV appeared meanwhile.
	method="${tag%%__*}"; rest="${tag#*__}"
	arch="${rest%%__*}"; rest="${rest#*__}"
	mode="${rest%%__*}"; rate="${rest##*__p}"

	echo ">>> $HOST claimed $tag"
	if EXP_METHODS="$method" EXP_ARCHS="$arch" EXP_MODES="$mode" EXP_RATES="$rate" \
	   EXP_LIMIT="$LIMIT" EXP_TIMEOUT="$TIMEOUT" EXP_SKIP_FILE="$SKIP_FILE" \
	   ./prune-experiment/recreate.sh --skip-existing; then
		worked=$((worked+1))
		echo "    $tag finished"
	else
		echo "    $tag recreate.sh exited non-zero (see $OUT/logs/$tag.log)"
	fi
	# Leave the lock in place as a done-marker; cell_done() gates reruns.
done < <(tags)

echo "worker $HOST drained the queue; ran $worked cell(s) this pass"
