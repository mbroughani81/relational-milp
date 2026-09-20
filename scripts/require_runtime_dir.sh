# Sourced by entry scripts to enforce that RUNTIME_DIR is set (fail-fast).
#
# RUNTIME_DIR is the single base directory for all generated/downloaded content
# (data fixtures, third_party checkouts, build artifacts). It is required so a
# misconfigured run stops immediately instead of writing to the wrong place.
#
# Usage (from a script that has already computed REPO_ROOT):
#   . "$REPO_ROOT/scripts/require_runtime_dir.sh"
# `exit` here aborts the sourcing script, which is the intended behaviour.
if [ -z "${RUNTIME_DIR:-}" ]; then
	echo "ERROR: RUNTIME_DIR is not set. Export it to the base directory for" >&2
	echo "       generated/downloaded content before running, e.g.:" >&2
	echo '         export RUNTIME_DIR="$PWD/runtime"' >&2
	exit 1
fi
