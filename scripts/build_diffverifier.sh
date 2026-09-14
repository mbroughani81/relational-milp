#!/usr/bin/env bash
#
# Build the ReluDiff and NeuroDiff verifier binaries from the NeuroDiff
# ASE-2020 artifact, plus their OpenBLAS 0.3.6 dependency.
#
# The artifact and OpenBLAS 0.3.6 predate modern GCC defaults, so this script
# applies the flags needed to build on GCC 14/15:
#   * OpenBLAS: demote -Wincompatible-pointer-types / -Wimplicit-function-declaration
#     / -Wint-conversion / -Wimplicit-int back to warnings (GCC 14+ made them errors).
#   * delta_network_test: the same demotions, plus -fcommon (GCC 10+ defaults to
#     -fno-common, which turns the artifact's header-defined `lock` global into a
#     multiple-definition link error).
# An rpath to the OpenBLAS prefix is baked into the binaries so they run without
# LD_LIBRARY_PATH set (e.g. from benchmarks.run_diffverifier's subprocess).
#
# Usage:
#   scripts/build_diffverifier.sh [ARTIFACT_DIR] [MAX_THREAD] [OPENBLAS_PREFIX]
# Defaults:
#   ARTIFACT_DIR   = third_party/NeuroDiff-ASE2020-Artifact
#   MAX_THREAD     = 4
#   OPENBLAS_PREFIX= $HOME/.local
set -euo pipefail

ARTIFACT_DIR="${1:-third_party/NeuroDiff-ASE2020-Artifact}"
MAX_THREAD="${2:-4}"
PREFIX="${3:-$HOME/.local}"

WNO="-Wno-incompatible-pointer-types -Wno-implicit-function-declaration"
WNO+=" -Wno-int-conversion -Wno-implicit-int"

if [[ ! -d "$ARTIFACT_DIR/DiffNN-Code" ]]; then
	echo "artifact not found at $ARTIFACT_DIR; clone it first:" >&2
	echo "  git clone https://github.com/pauls658/NeuroDiff-ASE2020-Artifact $ARTIFACT_DIR" >&2
	exit 1
fi

ARTIFACT_DIR="$(cd "$ARTIFACT_DIR" && pwd)"
DIFFNN="$ARTIFACT_DIR/DiffNN-Code"

# 1. OpenBLAS 0.3.6 (skip if already installed to PREFIX).
if [[ ! -f "$PREFIX/lib/libopenblas.so.0" ]]; then
	echo "=== building OpenBLAS 0.3.6 -> $PREFIX ==="
	mkdir -p "$PREFIX/lib" "$PREFIX/include"
	workdir="$(mktemp -d)"
	if [[ -f "$ARTIFACT_DIR/v0.3.6.zip" ]]; then
		unzip -q "$ARTIFACT_DIR/v0.3.6.zip" -d "$workdir"
	else
		curl -sL -o "$workdir/openblas.tar.gz" \
			https://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
		tar xzf "$workdir/openblas.tar.gz" -C "$workdir"
	fi
	pushd "$workdir/OpenBLAS-0.3.6" >/dev/null
	# delta_network_test only uses cblas_sgemm (single-precision BLAS) and needs
	# no LAPACK, so build BLAS-only: NO_LAPACK=1 NOFORTRAN=1 drops the Fortran
	# LAPACK (and thus any libgfortran runtime dependency), keeping the binaries
	# portable to servers without gfortran. Build the library targets explicitly
	# (libs shared) rather than the default `all`, which also runs OpenBLAS
	# 0.3.6's self-test suite; that test driver segfaults under modern GCC/glibc
	# and would abort the build before `make install`.
	make -j"$(nproc)" USE_THREAD=1 NO_LAPACK=1 NOFORTRAN=1 \
		COMMON_OPT="-O2 $WNO -Wno-implicit-int" libs shared
	make PREFIX="$PREFIX" NO_LAPACK=1 NOFORTRAN=1 install
	popd >/dev/null
	rm -rf "$workdir"
else
	echo "=== OpenBLAS already present at $PREFIX/lib ==="
fi

# 2. Ensure the makefile links against PREFIX with an rpath.
if ! grep -q -- "-rpath,$PREFIX/lib" "$DIFFNN/makefile"; then
	sed -i "s|^LDFLAGS= -lopenblas|LDFLAGS= -L$PREFIX/lib -Wl,-rpath,$PREFIX/lib -lopenblas|" \
		"$DIFFNN/makefile"
fi

# 3. Build both binaries.
export C_INCLUDE_PATH="$PREFIX/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$PREFIX/lib:${LIBRARY_PATH:-}"
CFLAGS_COMMON="-DMAX_THREAD=$MAX_THREAD $WNO -fcommon"

pushd "$DIFFNN" >/dev/null
echo "=== building pure ReluDiff (MAX_THREAD=$MAX_THREAD) ==="
CFLAGS="$CFLAGS_COMMON" make clean all
cp delta_network_test reludiff

echo "=== building full NeuroDiff (MAX_THREAD=$MAX_THREAD) ==="
CFLAGS="$CFLAGS_COMMON" make clean lineqall extravarssym all
cp delta_network_test neurodiff
popd >/dev/null

echo
echo "Built:"
echo "  $DIFFNN/reludiff"
echo "  $DIFFNN/neurodiff"
echo
echo "Run with, e.g.:"
echo "  python3 -m benchmarks.run_diffverifier --tool neurodiff \\"
echo "    --binary $DIFFNN/neurodiff \\"
echo "    --suite-options networks=mnist_relu_3_100 --suite-options modes=global \\"
echo "    --suite-options perturbation=prune --suite-options sparsity=0.3 \\"
echo "    --suite-options limit=100 --suite-options timeout=60"
