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
  runner uses the high-level Python API and writes per-instance configs/results
  under `artifacts/abcrown_instances/`.

Download the original ReluDiff MNIST networks and the paper's 100 test inputs:

```bash
python3 scripts/download_mnist_reludiff_nnets.py
```

The downloader reads the files from `DiffNN-Code/nnet` in the official
ReluDiff artifact and validates the architectures before installing them under
`data/reludiff_mnist/`. In particular, `mnist_relu_3_100` must be
`784-100-100-100-10`; files with architecture `784-100-100-10-10` are rejected.

Check the installed architecture headers without downloading anything:

```bash
python3 scripts/download_mnist_reludiff_nnets.py --check-only
```

The checker reads the files from `data/reludiff_mnist/` by default. Use
`--output-dir PATH` when the `.nnet` files are stored elsewhere. It exits with
status 1 if a file is missing, malformed, or has the wrong architecture.

## Run benchmarks

Run the small smoke-test suite with Pyomo, using HiGHS solver:

```bash
python3 -m benchmarks.run_pyomo --suite sample --solver highs
```

Run the same suite through Pyomo, using Gurobi solver:

```bash
python3 -m benchmarks.run_pyomo --suite sample --solver gurobi
```

Tighten Pyomo ReLU pre-activation bounds with alpha-beta-CROWN before solving:

```bash
python3 -m benchmarks.run_pyomo \
  --suite sample \
  --solver highs \
  --bound-tightening abcrown
```

Run the direct Gurobi encoding:

```bash
python3 -m benchmarks.run_gurobi --suite sample
```

Run alpha-beta-CROWN:

```bash
python3 -m benchmarks.run_crown --suite sample --profile relu-kfsb
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

### CPLEX debug and presolve statistics

Capture structured CPLEX statistics without running a second diagnostic
presolve pass:

```bash
python3 -m benchmarks.run_pyomo \
  --suite mnist_reludiff \
  --solver cplex \
  --debug-out artifacts/cplex_debug.json \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
  --suite-options limit=10 \
  --suite-options timeout=60
```

The runner captures the log produced by the actual `solver.solve(...)` call.
`cplex_presolve.initial_time_sec` and `after_presolve` are parsed from the first
CPLEX presolve summary. `cplex_presolve.total_reported_time_sec` sums every
`Presolve time` entry because CPLEX may run additional presolve passes during
root processing. The reduced-MIP log reports total binary columns but not a
separate unfixed-binary count, so
`after_presolve.unfixed_binary_variables` is reported as `unavailable`. Add
`--debug` to print the same structured JSON to stdout and `--verbose` to also
print the raw CPLEX log.

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
- `mnist_reludiff`: compares the original ReluDiff `.nnet` MNIST models with
  their float16 quantized versions. For image label `c`, each instance verifies
  `|nn1(x)[c] - nn2(x)[c]| <= epsilon`; it does not take a maximum over all ten
  outputs. This matches the artifact's per-image target-output setup. It supports
  `networks`, `modes`, `limit`, `timeout`, `epsilon`, and `perturb` suite options.
- `synthetic`: deterministic random 2D ReLU networks with architecture
  `2-10-10-2`; compares a base network with a noisy perturbation at several
  epsilon values.
- `bigger_synthetic`: a larger deterministic synthetic suite with architecture
  `2-1000-1000-1000-2`. This is intended for stress testing and may be slow or
  memory intensive.
- `mnist`: compares trained MNIST model pairs from `models/nn_equivalence/` on
  small input boxes around MNIST test samples from `data/MNIST/`.
