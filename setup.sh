#!/usr/bin/env bash
#
# One-shot environment bootstrap for relational-milp.
#
# Goal: clone this repo on a fresh cluster node, run `./setup.sh`, and then
# `prune-experiment/recreate.sh` (or distillation/recreate.sh) runs the
# benchmarks end to end.
#
# It provisions everything that can be automated:
#
#   1. system packages   (build tools, Python 3.11 + venv/pip, gfortran, unzip)
#   2. a project venv     (.venv, built with Python 3.11) with the requirements
#   3. alpha-beta-CROWN (abcrown) + auto_LiRPA installed INTO that venv, so
#      `from abcrown import ABCrownSolver` and `from auto_LiRPA import ...` work
#   4. the ReluDiff/NeuroDiff C verifiers (OpenBLAS + delta_network_test)
#   5. the ReluDiff MNIST fixtures under data/reludiff_mnist/
#
# Why Python 3.11? The alpha-beta-CROWN / auto_LiRPA releases that expose the
# high-level API this repo uses (ABCrownSolver, ConfigBuilder, IOConstraints,
# input_vars, output_vars) pin `requires-python = ~=3.11.0`. They do NOT install
# on 3.12. setup.sh therefore builds the venv with a Python 3.11 interpreter,
# installing one via the deadsnakes PPA when the host only ships a newer Python.
#
# One verifier depends on an external, non-free piece that a script cannot
# install; setup.sh only *detects* it and prints guidance:
#
#   * milp_abcrown -> CPLEX (commercial, licensed). Point setup.sh at your
#                     install with CPLEX_HOME=/path/to/CPLEX_StudioXXXX and it
#                     will install the matching python bindings into the venv.
#                     (milp_abcrown also needs abcrown, which IS installed here.)
#
# recreate.sh detects whichever of the four verifiers are actually available and
# runs those, skipping the rest, so a partial environment still produces results.
#
# Usage:
#   ./setup.sh                 # full setup (CPU torch 2.11 + abcrown + auto_LiRPA)
#   ./setup.sh --no-torch      # skip torch/abcrown/auto_LiRPA (reludiff/neurodiff only)
#   ./setup.sh --skip-system   # don't touch apt (no sudo / already provisioned)
#   ./setup.sh --help
#
# Env overrides:
#   PYTHON_BIN      Python 3.11 interpreter to build the venv with
#                   (default: autodetect python3.11, else install via deadsnakes)
#   TORCH_INDEX_URL pip index for torch/torchvision wheels
#                   (default: https://download.pytorch.org/whl/cpu; set to a CUDA
#                   index such as https://download.pytorch.org/whl/cu124 on a GPU host)
#   ABCROWN_HOME    use this existing alpha-beta-CROWN checkout instead of cloning
#                   (its auto_LiRPA submodule must be present / initialized)
#   ABCROWN_REPO    alpha-beta-CROWN git remote (default: official Verified-Intelligence)
#   ABCROWN_COMMIT  pinned alpha-beta-CROWN commit to check out (default: validated pin)
#   CPLEX_HOME      CPLEX_Studio dir whose python bindings to install into .venv
#   SKIP_TORCH=1    same as --no-torch
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"
ARTIFACT_DIR="$REPO_ROOT/third_party/NeuroDiff-ASE2020-Artifact"
DIFFNN="$ARTIFACT_DIR/DiffNN-Code"
ARTIFACT_REPO="https://github.com/pauls658/NeuroDiff-ASE2020-Artifact"

# alpha-beta-CROWN (abcrown) + its auto_LiRPA submodule. Pinned to the commit
# this repo was validated against; override ABCROWN_COMMIT to move it.
ABCROWN_REPO="${ABCROWN_REPO:-https://github.com/Verified-Intelligence/alpha-beta-CROWN.git}"
ABCROWN_COMMIT="${ABCROWN_COMMIT:-e5c7e17bf0488843acb77b7519f59876717a49f4}"
ABCROWN_DIR_DEFAULT="$REPO_ROOT/third_party/alpha-beta-CROWN"

# torch/torchvision versions abcrown pins; pre-installed from a CPU index by
# default so a CPU cluster node does not pull ~2GB of CUDA wheels.
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

SKIP_SYSTEM=0
SKIP_TORCH="${SKIP_TORCH:-0}"
for arg in "$@"; do
	case "$arg" in
		--no-torch)    SKIP_TORCH=1 ;;
		--skip-system) SKIP_SYSTEM=1 ;;
		-h|--help)     sed -n '2,60p' "$0"; exit 0 ;;
		*) echo "unknown argument: $arg" >&2; exit 2 ;;
	esac
done

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ok:\033[0m %s\n' "$*"; }

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# --------------------------------------------------------------------------- #
# Pick the Python interpreter used to build the venv.
#
# abcrown/auto_LiRPA require Python 3.11, so the full setup needs a 3.11
# interpreter. --no-torch skips abcrown entirely, so any Python 3.10+ is fine
# there. PYTHON_BIN overrides the autodetection.
# --------------------------------------------------------------------------- #
PYTHON_BIN="${PYTHON_BIN:-}"

python_is_311() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; }

select_python() {
	if [ -n "$PYTHON_BIN" ]; then
		command -v "$PYTHON_BIN" >/dev/null 2>&1 || { warn "PYTHON_BIN=$PYTHON_BIN not found"; exit 2; }
		return
	fi
	if command -v python3.11 >/dev/null 2>&1; then
		PYTHON_BIN="python3.11"
	elif [ "$SKIP_TORCH" -eq 1 ] && command -v python3 >/dev/null 2>&1; then
		# reludiff/neurodiff-only: no abcrown, so a newer Python is acceptable.
		PYTHON_BIN="python3"
	else
		PYTHON_BIN=""   # install_system will provision 3.11 and re-select
	fi
}

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
python3.11 python3.11-venv python3.11-dev python3-pip git curl"
		return
	fi
	$SUDO apt-get update -qq
	DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
		build-essential gfortran unzip python3-pip git curl
	ok "base apt packages installed"

	# Ensure a Python 3.11 interpreter (+ venv/dev) unless --no-torch already
	# found a usable python3.
	if [ "$SKIP_TORCH" -eq 1 ] && [ -n "$PYTHON_BIN" ]; then
		ok "python for --no-torch: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
		return
	fi
	if ! command -v python3.11 >/dev/null 2>&1; then
		log "Installing Python 3.11 (abcrown/auto_LiRPA require it)"
		if ! apt-cache policy python3.11 2>/dev/null | grep -q Candidate:.*[0-9]; then
			DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq software-properties-common
			$SUDO add-apt-repository -y ppa:deadsnakes/ppa
			$SUDO apt-get update -qq
		fi
		DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
			python3.11 python3.11-venv python3.11-dev
	else
		DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
			python3.11-venv python3.11-dev || true
	fi
	command -v python3.11 >/dev/null 2>&1 && PYTHON_BIN="python3.11"
	ok "python3.11 available ($(python3.11 --version 2>&1))"
}

# --------------------------------------------------------------------------- #
# 2. Python venv + requirements (+ pinned CPU torch)
# --------------------------------------------------------------------------- #
install_python() {
	log "Python virtualenv + requirements"
	[ -n "$PYTHON_BIN" ] || { warn "no Python interpreter selected; cannot build venv"; exit 2; }
	if [ "$SKIP_TORCH" -ne 1 ] && ! python_is_311 "$PYTHON_BIN"; then
		warn "$PYTHON_BIN is $($PYTHON_BIN --version 2>&1), but abcrown/auto_LiRPA need 3.11."
		warn "install Python 3.11 or pass PYTHON_BIN=/path/to/python3.11 (or use --no-torch)."
		exit 2
	fi

	if [ ! -x "$PY" ]; then
		"$PYTHON_BIN" -m venv "$VENV"
		ok "created venv at ${VENV#$REPO_ROOT/} ($($PY --version 2>&1))"
	else
		ok "venv already exists at ${VENV#$REPO_ROOT/} ($($PY --version 2>&1))"
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
		# Pre-install the exact torch abcrown pins, from a CPU index by default,
		# so the later `pip install abcrown` finds it satisfied instead of
		# pulling the CUDA build. On a GPU host set TORCH_INDEX_URL to a cuXXX
		# index (or install the matching GPU wheel yourself first).
		"$PY" -m pip install --index-url "$TORCH_INDEX_URL" \
			"torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
		"$PY" -m pip install -r requirements.txt
	fi
	ok "python requirements installed"
}

# --------------------------------------------------------------------------- #
# 3. alpha-beta-CROWN (abcrown) + auto_LiRPA
#
# The abcrown wheel bundles the `abcrown`/`complete_verifier` packages but NOT
# auto_LiRPA (it is a git submodule), so both are installed: auto_LiRPA from the
# submodule, then abcrown itself.
# --------------------------------------------------------------------------- #
install_abcrown() {
	log "alpha-beta-CROWN (abcrown) + auto_LiRPA"
	if [ "$SKIP_TORCH" -eq 1 ]; then
		warn "torch not installed (--no-torch) -> abcrown/auto_LiRPA skipped"
		return
	fi

	local abcrown_dir
	if [ -n "${ABCROWN_HOME:-}" ]; then
		abcrown_dir="$ABCROWN_HOME"
		[ -d "$abcrown_dir" ] || { warn "ABCROWN_HOME=$abcrown_dir not a directory"; return; }
		ok "using existing checkout ABCROWN_HOME=$abcrown_dir"
	else
		abcrown_dir="$ABCROWN_DIR_DEFAULT"
		if [ ! -d "$abcrown_dir/.git" ]; then
			git clone "$ABCROWN_REPO" "$abcrown_dir"
			git -C "$abcrown_dir" checkout "$ABCROWN_COMMIT"
		else
			ok "alpha-beta-CROWN checkout already present"
		fi
		git -C "$abcrown_dir" submodule update --init --recursive
	fi

	if [ ! -d "$abcrown_dir/auto_LiRPA/auto_LiRPA" ] && [ ! -f "$abcrown_dir/auto_LiRPA/pyproject.toml" ]; then
		warn "auto_LiRPA submodule not initialized under $abcrown_dir; run:"
		warn "  git -C $abcrown_dir submodule update --init --recursive"
		return
	fi

	"$PY" -m pip install "$abcrown_dir/auto_LiRPA"
	"$PY" -m pip install "$abcrown_dir"

	if "$PY" -c "from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm" >/dev/null 2>&1; then
		ok "auto_LiRPA importable"
	else
		warn "auto_LiRPA still not importable after install"
	fi
	if "$PY" -c "from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints, input_vars, output_vars" >/dev/null 2>&1; then
		ok "abcrown high-level API importable"
	else
		warn "abcrown not importable after install -> abcrown method will be skipped by recreate.sh"
	fi
}

# --------------------------------------------------------------------------- #
# 4. ReluDiff / NeuroDiff C verifiers
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
# 5. ReluDiff MNIST fixtures
# --------------------------------------------------------------------------- #
download_data() {
	log "ReluDiff MNIST fixtures"
	"$PY" scripts/download_mnist_reludiff_nnets.py --artifact-dir "$ARTIFACT_DIR"
	ok "data/reludiff_mnist populated"
}

# --------------------------------------------------------------------------- #
# 6. CPLEX (milp_abcrown) - detect / optionally install python bindings
# --------------------------------------------------------------------------- #
setup_cplex() {
	log "CPLEX (for milp_abcrown)"
	# milp_abcrown solves the MILP with CPLEX. benchmarks.run_pyomo drives it
	# through the file backend by default (CPLEX_BACKEND=auto), which needs the
	# full-edition `cplex` CLI on PATH -- NOT the PyPI `cplex` wheel, which is the
	# size-capped Community Edition (max 1000 vars/constraints -> CPLEX Error 1016
	# on these models). So we locate the CLI from a CPLEX Studio install and put
	# it on PATH; we deliberately do NOT `pip install cplex`.
	local cli=""
	if [ -n "${CPLEX_HOME:-}" ]; then
		cli="$(ls "$CPLEX_HOME"/cplex/bin/*/cplex 2>/dev/null | head -1 || true)"
		[ -z "$cli" ] && warn "CPLEX_HOME=$CPLEX_HOME set but no cplex/bin/*/cplex under it"
	fi
	[ -z "$cli" ] && cli="$(command -v cplex 2>/dev/null || true)"

	if [ -n "$cli" ] && [ -x "$cli" ]; then
		if echo quit | "$cli" 2>&1 | grep -qi "Community Edition"; then
			warn "cplex at $cli is Community Edition (size-capped) -> milp_abcrown will"
			warn "  fail on real models. Point CPLEX_HOME at a full CPLEX Studio install."
		elif [ "$cli" = "/usr/local/bin/cplex" ]; then
			ok "full CPLEX CLI on PATH ($cli)"
			return
		elif $SUDO ln -sf "$cli" /usr/local/bin/cplex 2>/dev/null; then
			ok "linked full CPLEX CLI -> /usr/local/bin/cplex ($cli)"
			return
		else
			ok "full CPLEX CLI at $cli"
			warn "  could not symlink into /usr/local/bin; export CPLEX_HOME (or add it"
			warn "  to PATH) in the shell that runs recreate.sh so the runner finds it."
			return
		fi
	fi
	warn "no full CPLEX CLI found -> milp_abcrown will be skipped by recreate.sh."
	warn "  install a licensed CPLEX Studio, then re-run with CPLEX_HOME=/path/to/CPLEX_StudioXXXX"
	warn "  (do NOT rely on 'pip install cplex' -- that is the size-capped Community Edition)"
}

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
summary() {
	log "Environment summary"
	local have_diff=no have_cplex=no have_abcrown=no have_lirpa=no cplex_cli=""
	[ -x "$DIFFNN/reludiff" ] && [ -x "$DIFFNN/neurodiff" ] && have_diff=yes
	# Report the CPLEX the runner will actually use (its resolved full-edition
	# CLI), not a bare `import cplex` (which the community wheel also satisfies).
	cplex_cli="$("$PY" -c "from benchmarks.run_pyomo import resolve_cplex_executable as r; print(r() or '')" 2>/dev/null || true)"
	[ -n "$cplex_cli" ] && have_cplex=yes
	"$PY" -c "from abcrown import ABCrownSolver" >/dev/null 2>&1 && have_abcrown=yes
	"$PY" -c "from auto_LiRPA import BoundedModule" >/dev/null 2>&1 && have_lirpa=yes

	printf '  %-20s %s\n' "python:"            "$($PY --version 2>&1)"
	printf '  %-20s %s\n' "reludiff/neurodiff:" "$have_diff"
	printf '  %-20s %s\n' "abcrown:"           "$have_abcrown   (abcrown method + milp_abcrown tightening)"
	printf '  %-20s %s\n' "auto_LiRPA:"        "$have_lirpa"
	printf '  %-20s %s\n' "cplex (full CLI):"  "$have_cplex   (milp_abcrown)${cplex_cli:+ -> $cplex_cli}"
	echo
	echo "Next:"
	echo "  ./prune-experiment/recreate.sh --light   # quick smoke run"
	echo "  ./prune-experiment/recreate.sh           # full sweep"
}

select_python
install_system
install_python
install_abcrown
build_diffverifier
download_data
setup_cplex
summary
