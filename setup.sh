#!/usr/bin/env bash
#
# One-shot environment bootstrap for relational-milp.
#
# Goal: clone this repo on a fresh server, run `./setup.sh`, and then
# `prune-experiment/recreate.sh` (or distillation/recreate.sh) runs the
# benchmarks end to end.
#
# It provisions everything that can be automated:
#
#   1. system packages   (build tools, python venv/pip, gfortran, unzip)
#   2. a project venv     (.venv) with the Python requirements
#   3. the ReluDiff/NeuroDiff C verifiers (OpenBLAS + delta_network_test)
#   4. the ReluDiff MNIST fixtures under data/reludiff_mnist/
#
# Two verifiers depend on external pieces that are NOT freely installable and
# therefore cannot be provisioned from a script; setup.sh only *detects* them
# and prints guidance:
#
#   * milp_abcrown -> CPLEX (commercial, licensed). Point setup.sh at your
#                     install with CPLEX_HOME=/path/to/CPLEX_StudioXXXX and it
#                     will install the matching python bindings into the venv.
#   * abcrown      -> alpha-beta-CROWN. Provide its checkout via ABCROWN_HOME so
#                     `from abcrown import ABCrownSolver` resolves.
#
# recreate.sh detects whichever of the four verifiers are actually available and
# runs those, skipping the rest, so a partial environment still produces results.
#
# Usage:
#   ./setup.sh                 # full setup (installs CPU torch for abcrown)
#   ./setup.sh --no-torch      # skip torch/torchvision (reludiff/neurodiff only)
#   ./setup.sh --skip-system   # don't touch apt (no sudo / already provisioned)
#   ./setup.sh --help
#
# Env overrides:
#   CPLEX_HOME    CPLEX_Studio dir whose python bindings to install into .venv
#   ABCROWN_HOME  alpha-beta-CROWN checkout to expose on the venv path
#   SKIP_TORCH=1  same as --no-torch
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"
ARTIFACT_DIR="$REPO_ROOT/third_party/NeuroDiff-ASE2020-Artifact"
DIFFNN="$ARTIFACT_DIR/DiffNN-Code"
ARTIFACT_REPO="https://github.com/pauls658/NeuroDiff-ASE2020-Artifact"

SKIP_SYSTEM=0
SKIP_TORCH="${SKIP_TORCH:-0}"
for arg in "$@"; do
	case "$arg" in
		--no-torch)    SKIP_TORCH=1 ;;
		--skip-system) SKIP_SYSTEM=1 ;;
		-h|--help)     sed -n '2,45p' "$0"; exit 0 ;;
		*) echo "unknown argument: $arg" >&2; exit 2 ;;
	esac
done

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ok:\033[0m %s\n' "$*"; }

# --------------------------------------------------------------------------- #
# 1. System packages
# --------------------------------------------------------------------------- #
install_system() {
	log "System packages"
	if [ "$SKIP_SYSTEM" -eq 1 ]; then
		ok "skipped (--skip-system)"
		return
	fi
	if ! command -v apt-get >/dev/null 2>&1; then
		warn "no apt-get; install these yourself: build-essential gfortran unzip \
python3-venv python3-pip git curl"
		return
	fi
	local SUDO=""
	[ "$(id -u)" -ne 0 ] && SUDO="sudo"
	$SUDO apt-get update -qq
	DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
		build-essential gfortran unzip python3-venv python3-pip git curl
	ok "apt packages installed"
}

# --------------------------------------------------------------------------- #
# 2. Python venv + requirements
# --------------------------------------------------------------------------- #
install_python() {
	log "Python virtualenv + requirements"
	if [ ! -x "$PY" ]; then
		python3 -m venv "$VENV"
		ok "created venv at ${VENV#$REPO_ROOT/}"
	else
		ok "venv already exists at ${VENV#$REPO_ROOT/}"
	fi
	"$PY" -m pip install --upgrade pip >/dev/null

	if [ "$SKIP_TORCH" -eq 1 ]; then
		# Install everything except torch/torchvision (the reludiff/neurodiff
		# path and the data tooling need none of it).
		local tmp_req
		tmp_req="$(mktemp)"
		grep -viE '^(torch|torchvision)([<>=~! ]|$)' requirements.txt >"$tmp_req"
		"$PY" -m pip install -r "$tmp_req"
		rm -f "$tmp_req"
		warn "torch/torchvision skipped (--no-torch): abcrown/milp_abcrown \
bound-tightening will be unavailable"
	else
		# CPU torch wheels; on a CUDA host, install the matching GPU wheel
		# yourself before/after this and it will be kept.
		"$PY" -m pip install -r requirements.txt
	fi
	ok "python requirements installed"
}

# --------------------------------------------------------------------------- #
# 3. ReluDiff / NeuroDiff C verifiers
# --------------------------------------------------------------------------- #
build_diffverifier() {
	log "ReluDiff / NeuroDiff verifiers"
	if [ -x "$DIFFNN/reludiff" ] && [ -x "$DIFFNN/neurodiff" ]; then
		ok "binaries already built"
		return
	fi
	if [ ! -d "$DIFFNN" ]; then
		git clone --depth 1 "$ARTIFACT_REPO" "$ARTIFACT_DIR"
	fi
	bash scripts/build_diffverifier.sh
	if [ -x "$DIFFNN/reludiff" ] && [ -x "$DIFFNN/neurodiff" ]; then
		ok "reludiff + neurodiff built"
	else
		warn "diffverifier build did not produce both binaries; reludiff/neurodiff \
will be skipped by recreate.sh"
	fi
}

# --------------------------------------------------------------------------- #
# 4. ReluDiff MNIST fixtures
# --------------------------------------------------------------------------- #
download_data() {
	log "ReluDiff MNIST fixtures"
	"$PY" scripts/download_mnist_reludiff_nnets.py --artifact-dir "$ARTIFACT_DIR"
	ok "data/reludiff_mnist populated"
}

# --------------------------------------------------------------------------- #
# 5. CPLEX (milp_abcrown) - detect / optionally install python bindings
# --------------------------------------------------------------------------- #
setup_cplex() {
	log "CPLEX (for milp_abcrown)"
	if "$PY" -c "import cplex" >/dev/null 2>&1; then
		ok "cplex python bindings importable in venv"
		return
	fi
	if [ -n "${CPLEX_HOME:-}" ]; then
		local setup_py="$CPLEX_HOME/python/setup.py"
		if [ -f "$setup_py" ]; then
			( cd "$CPLEX_HOME/python" && "$PY" setup.py install ) \
				&& ok "installed cplex bindings from $CPLEX_HOME" && return
		fi
		# Newer studios ship a wheel instead of setup.py.
		"$PY" -m pip install cplex >/dev/null 2>&1 \
			&& ok "installed cplex wheel" && return
		warn "CPLEX_HOME=$CPLEX_HOME set but could not install bindings from it"
	fi
	warn "CPLEX not available -> milp_abcrown will be skipped by recreate.sh."
	warn "  install a licensed CPLEX, then re-run with CPLEX_HOME=/path/to/CPLEX_StudioXXXX"
}

# --------------------------------------------------------------------------- #
# 6. alpha-beta-CROWN (abcrown) - detect / expose on path
# --------------------------------------------------------------------------- #
setup_abcrown() {
	log "alpha-beta-CROWN (for abcrown + milp_abcrown bound tightening)"
	if [ "$SKIP_TORCH" -eq 1 ]; then
		warn "torch not installed (--no-torch) -> abcrown unavailable"
		return
	fi
	local extra=""
	[ -n "${ABCROWN_HOME:-}" ] && extra="PYTHONPATH=$ABCROWN_HOME"
	if env $extra "$PY" -c "from abcrown import ABCrownSolver" >/dev/null 2>&1; then
		ok "abcrown importable${ABCROWN_HOME:+ (via ABCROWN_HOME=$ABCROWN_HOME)}"
		if [ -n "${ABCROWN_HOME:-}" ]; then
			# Persist the path so recreate.sh's venv python can import it.
			echo "$ABCROWN_HOME" > "$VENV/lib/abcrown.pth" 2>/dev/null \
				|| warn "could not persist ABCROWN_HOME to the venv; export PYTHONPATH yourself"
		fi
		return
	fi
	warn "abcrown not importable -> abcrown method will be skipped by recreate.sh."
	warn "  provide an alpha-beta-CROWN checkout via ABCROWN_HOME=/path/to/alpha-beta-CROWN"
}

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
summary() {
	log "Environment summary"
	local have_diff=no have_cplex=no have_abcrown=no
	[ -x "$DIFFNN/reludiff" ] && [ -x "$DIFFNN/neurodiff" ] && have_diff=yes
	"$PY" -c "import cplex" >/dev/null 2>&1 && have_cplex=yes
	"$PY" -c "from abcrown import ABCrownSolver" >/dev/null 2>&1 && have_abcrown=yes

	printf '  %-14s %s\n' "reludiff/neurodiff:" "$have_diff"
	printf '  %-14s %s\n' "cplex:"    "$have_cplex   (milp_abcrown)"
	printf '  %-14s %s\n' "abcrown:"  "$have_abcrown   (abcrown + milp_abcrown tightening)"
	echo
	echo "Next:"
	echo "  ./prune-experiment/recreate.sh --light   # quick smoke run"
	echo "  ./prune-experiment/recreate.sh           # full sweep"
}

install_system
install_python
build_diffverifier
download_data
setup_cplex
setup_abcrown
summary
