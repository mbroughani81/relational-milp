#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --no-fix-stable-relu-binaries \
  --debug \
  --debug-out phase-analysis/no-fix-2-512-global-limit-10.json \
  --suite-options networks=mnist_relu_2_512 \
  --suite-options modes=global \
  --suite-options timeout=120 \
  --suite-options limit=3 \
  --csv phase-analysis/no-fix-2-512-global-limit-10.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --no-fix-stable-relu-binaries \
  --debug \
  --debug-out phase-analysis/no-fix-4-1024-3pixel-limit-10.json \
  --suite-options networks=mnist_relu_4_1024 \
  --suite-options modes=three_pixel \
  --suite-options timeout=120 \
  --suite-options limit=10 \
  --csv phase-analysis/no-fix-4-1024-3pixel-limit-10.csv


