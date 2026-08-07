#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --debug \
  --debug-out fixing-binary-vars-experiment/fix-2-512-3pixel.json \
  --suite-options networks=mnist_relu_2_512 \
  --suite-options modes=three_pixel \
  --suite-options timeout=60 \
  --suite-options limit=10 \
  --csv fixing-binary-vars-experiment/fix-2-512-3pixel.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --no-fix-stable-relu-binaries \
  --debug \
  --debug-out fixing-binary-vars-experiment/no-fix-2-512-3pixel.json \
  --suite-options networks=mnist_relu_2_512 \
  --suite-options modes=three_pixel \
  --suite-options timeout=60 \
  --suite-options limit=100 \
  --csv fixing-binary-vars-experiment/no-fix-2-512-3pixel.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --debug \
  --debug-out fixing-binary-vars-experiment/fix-3-100-3pixel.json \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=three_pixel \
  --suite-options timeout=60 \
  --suite-options limit=100 \
  --csv fixing-binary-vars-experiment/fix-3-100-3pixel.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --no-fix-stable-relu-binaries \
  --debug \
  --debug-out fixing-binary-vars-experiment/no-fix-3-100-3pixel.json \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=three_pixel \
  --suite-options timeout=60 \
  --suite-options limit=100 \
  --csv fixing-binary-vars-experiment/no-fix-3-100-3pixel.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --debug \
  --debug-out fixing-binary-vars-experiment/fix-3-100-global.json \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
  --suite-options timeout=60 \
  --suite-options limit=100 \
  --csv fixing-binary-vars-experiment/fix-3-100-global.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --debug \
  --debug-out fixing-binary-vars-experiment/fix-4-1024-3pixel.json \
  --suite-options networks=mnist_relu_4_1024 \
  --suite-options modes=three_pixel \
  --suite-options timeout=60 \
  --suite-options limit=100 \
  --csv fixing-binary-vars-experiment/fix-4-1024-3pixel.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --no-fix-stable-relu-binaries \
  --debug \
  --debug-out fixing-binary-vars-experiment/no-fix-3-100-global.json \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
  --suite-options timeout=60 \
  --suite-options limit=100 \
  --csv fixing-binary-vars-experiment/no-fix-3-100-global.csv

python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --bound-tightening abcrown \
  --no-fix-stable-relu-binaries \
  --debug \
  --debug-out fixing-binary-vars-experiment/no-fix-4-1024-3pixel.json \
  --suite-options networks=mnist_relu_4_1024 \
  --suite-options modes=three_pixel \
  --suite-options timeout=60 \
  --suite-options limit=100 \
  --csv fixing-binary-vars-experiment/no-fix-4-1024-3pixel.csv
