# Relational-MILP for neural-network equivalence

## Setup

Use Python 3.10 or newer from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

External solver/runtime requirements:

- HiGHS: install `highspy` with the requirements above. This is used by Pyomo
  when running `--solver highs`.
- Gurobi: install Gurobi, configure a valid license, and install `gurobipy`.
  This is needed for the direct Gurobi runner and for Pyomo with
  `--solver gurobi`.
- alpha-beta-CROWN: install alpha-beta-CROWN in the same Python
  environment so that `from abcrown import ABCrownSolver` works. The CROWN
  runner exports ONNX/VNNLIB artifacts under `artifacts/abcrown_instances/`.

## Run benchmarks

Run the small smoke-test suite with Pyomo, using HiGHS solver:

```bash
python3 -m benchmarks.run_pyomo --suite sample --solver highs
```

Run the same suite through Pyomo, using Gurobi solver:

```bash
python3 -m benchmarks.run_pyomo --suite sample --solver gurobi
```

Run the direct Gurobi encoding:

```bash
python3 -m benchmarks.run_gurobi --suite sample
```

Run alpha-beta-CROWN:

```bash
python3 -m benchmarks.run_crown --suite sample --profile beta_strong
```

List the available alpha-beta-CROWN profiles:

```bash
python3 -m benchmarks.run_crown --list-profiles
```

Suite-specific options are passed with repeated `--suite-options KEY=VALUE`
arguments. For example, to run a small ReluDiff MNIST subset:

```bash
python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver highs \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global,three_pixel \
  --suite-options limit=3 \
  --suite-options timeout=10
```

Redirect stdout to save benchmark results:

```bash
python3 -m benchmarks.run_pyomo --suite synthetic --solver highs > synthetic_highs.csv
python3 summarize_out_csv.py synthetic_highs.csv
```

Save backend solver logs and per-direction wall-clock timings while keeping CSV
results on stdout:

```bash
python3 -m benchmarks.run_pyomo \
  --suite sample \
  --solver highs \
  --solver-log-dir artifacts/solver_logs/sample_highs
```

## Current benchmark suites

- `sample`: three tiny 2-input instances. Includes slightly different networks
  at two epsilon values and an identical-network case. This is the fastest
  correctness smoke test.
- `mnist_reludiff`: compares ReluDiff `.nnet` MNIST models with their float16
  quantized versions. It supports `networks`, `modes`, `limit`, `timeout`,
  `epsilon`, and `perturb` suite options.
- `synthetic`: deterministic random 2D ReLU networks with architecture
  `2-10-10-2`; compares a base network with a noisy perturbation at several
  epsilon values.
- `bigger_synthetic`: a larger deterministic synthetic suite with architecture
  `2-1000-1000-1000-2`. This is intended for stress testing and may be slow or
  memory intensive.
- `mnist`: compares trained MNIST model pairs from `models/nn_equivalence/` on
  small input boxes around MNIST test samples from `data/MNIST/`.
