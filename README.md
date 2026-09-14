# Relational-MILP for neural-network equivalence

## Setup

### One-shot setup (recommended)

Clone the repo on a fresh server and run:

```bash
./setup.sh
```

This provisions everything that can be automated: system build tools, a project
`.venv` with the Python requirements, the ReluDiff/NeuroDiff C verifiers
(OpenBLAS + `delta_network_test`), and the ReluDiff MNIST fixtures under
`data/reludiff_mnist/`. Afterwards `prune-experiment/recreate.sh` runs end to end.

Two verifiers rely on external, non-free pieces that a script cannot install; it
only detects them and prints guidance:

- `milp_abcrown` needs a licensed **CPLEX**. Point setup at your install with
  `CPLEX_HOME=/path/to/CPLEX_StudioXXXX ./setup.sh` to install its python
  bindings into the venv.
- `abcrown` (and `milp_abcrown`'s bound tightening) needs **alpha-beta-CROWN**.
  Expose a checkout with `ABCROWN_HOME=/path/to/alpha-beta-CROWN ./setup.sh`.

`recreate.sh` runs whichever of the four verifiers are available and skips the
rest, so a partial environment still produces results. Useful flags:
`./setup.sh --no-torch` (skip torch, i.e. reludiff/neurodiff only) and
`./setup.sh --skip-system` (don't touch apt).

### Manual setup

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

Run ReluDiff / NeuroDiff (the `delta_network_test` C verifiers) on the same
`mnist_reludiff` instances as the MILP and CROWN runners:

```bash
python3 -m benchmarks.run_diffverifier \
  --tool neurodiff \
  --binary /path/to/delta_network_test \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
  --suite-options perturbation=prune \
  --suite-options sparsity=0.3 \
  --suite-options limit=100 \
  --suite-options timeout=60
```

The binary is the `delta_network_test` executable compiled from the NeuroDiff
ASE-2020 artifact; the same interface yields pure ReluDiff or full NeuroDiff
depending on the build flags (`--tool` only labels the output). The runner
serializes each pair's second network to a temporary `.nnet` (reusing the base
network's normalization header), maps `global`/`three_pixel` onto the ASE tool's
`-p`/`-x 3` flags and `sample_index` onto property ids `400-499`, parses
`No adv!` (verified -> `unsat`) / `adv found` (counterexample -> `sat`), and
emits the same CSV columns as the CROWN runner plus `num_splits` and
`tool_time_sec`. Pass `--binary` or set `DIFFVERIFIER_BINARY`.

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
`after_presolve.time_sec` and `after_presolve.binary_variables` are parsed from
the first CPLEX presolve summary. The time is the `Presolve time` value for
that summary, and the binary count is the number of binary columns in the
reduced MIP. Add `--debug` to print the same structured JSON to stdout and
`--verbose` to also print the raw CPLEX log.

Save backend solver logs and per-direction wall-clock timings while keeping CSV
results on stdout:

```bash
python3 -m benchmarks.run_pyomo \
  --suite sample \
  --solver highs \
  --solver-log-dir artifacts/solver_logs/sample_highs
```

## Current benchmark suites

- `mnist_reludiff`: compares the original ReluDiff `.nnet` MNIST models with a
  float16-quantized or magnitude-pruned copy of the *same* architecture. For
  image label `c`, each instance verifies `|nn1(x)[c] - nn2(x)[c]| <= epsilon`.
  Supports `networks`, `modes`, `limit`, `timeout`, `epsilon`, `perturb`,
  `perturbation`, and `sparsity` suite options.
- `distillation`: Hinton-style knowledge-distillation teacher/student pairs.
  Property A is numerical logit equivalence on the labeled class:
  `|z_T(x)[c] - z_S(x)[c]| <= epsilon`. Uses the same ReluDiff 100-image /
  3-pixel fixtures as `mnist_reludiff`. Supports `pairs`, `tiers`, `modes`,
  `limit`, `timeout`, `epsilon`, and `perturb`.

  Two experimental tiers:

  - **Tier A (same architecture)** — `kd_a1` (`784-64-32-10`→same), `kd_a2`
    (`784-128-64-10`→same). Fair head-to-head with ReluDiff / NeuroDiff.
  - **Tier B (different architecture)** — `kd_1` / `kd_2` / `kd_3`. Stress test
    for architecture-flexible verifiers (Relational-MILP, ab-CROWN); ReluDiff /
    NeuroDiff are N/A.

Train pairs:

```bash
# Tier A (ReluDiff/NeuroDiff comparable)
python3 -m training.distill_mnist --pair-id kd_a1 --force
python3 -m training.distill_mnist --pair-id kd_a2 --force

# Tier B (diff-arch; MILP/CROWN only)
python3 -m training.distill_mnist --pair-id kd_1 --force
```

Run Relational-MILP (epsilon from each pair's `metadata.json` logit-gap stats):

```bash
# Tier A fair comparison
python3 -m benchmarks.run_pyomo \
  --suite distillation \
  --solver cplex \
  --suite-options tiers=A \
  --suite-options modes=three_pixel \
  --suite-options epsilon=2.0 \
  --suite-options limit=100 \
  --suite-options timeout=30

# Tier B capability stress
python3 -m benchmarks.run_pyomo \
  --suite distillation \
  --solver cplex \
  --suite-options pairs=kd_1 \
  --suite-options modes=three_pixel \
  --suite-options epsilon=2.0 \
  --suite-options limit=100 \
  --suite-options timeout=30
```

Run ReluDiff / NeuroDiff on Tier A only:

```bash
python3 -m benchmarks.run_diffverifier \
  --suite distillation \
  --tool neurodiff \
  --binary /path/to/delta_network_test \
  --suite-options pairs=kd_a1 \
  --suite-options modes=three_pixel \
  --suite-options epsilon=2.0 \
  --suite-options limit=100 \
  --suite-options timeout=30
```

Other historically documented suites (`sample`, `synthetic`, `bigger_synthetic`,
`mnist`) may not be present in the current tree.