# Relational-MILP for neural-network equivalence

## Setup

### One-shot setup (recommended)

All generated/downloaded content (data fixtures, `third_party` checkouts, build
artifacts) lives under a single base directory set by the **required**
`RUNTIME_DIR` environment variable, so the repo base stays clean. The scripts
and Python entrypoints fail fast if it is unset. Export it once (keep it in the
same directory for later runs — e.g. `recreate.py`, `worker.py`):

```bash
export RUNTIME_DIR="$PWD/runtime"
```

`RUNTIME_DIR` may point anywhere (a shared volume, a scratch disk); only the
`.venv` stays at the repo root, by convention.

Clone the repo on a fresh server and run:

```bash
export RUNTIME_DIR="$PWD/runtime"
./setup.sh
```

This provisions everything that can be automated: system build tools, a
**Python 3.11** project `.venv` with the Python requirements, **alpha-beta-CROWN
(`abcrown`) and `auto_LiRPA`** installed into that venv, the ReluDiff/NeuroDiff C
verifiers (OpenBLAS + `delta_network_test`), and the ReluDiff MNIST fixtures
under `runtime/data/reludiff_mnist/`. Afterwards `prune-experiment/recreate.py` runs end
to end.

Python 3.11 is required because the alpha-beta-CROWN / `auto_LiRPA` releases that
expose the high-level API this repo uses (`ABCrownSolver`, `ConfigBuilder`,
`IOConstraints`, `input_vars`, `output_vars`) pin `requires-python = ~=3.11.0`.
`setup.sh` installs a 3.11 interpreter via the deadsnakes PPA when the host only
ships a newer Python; pass `PYTHON_BIN=/path/to/python3.11` to use your own.

alpha-beta-CROWN is cloned to `runtime/third_party/alpha-beta-CROWN` (pinned commit,
override with `ABCROWN_COMMIT=`) with its `auto_LiRPA` submodule, and both are
`pip install`ed into the venv. To reuse an existing checkout instead, point setup
at it with `ABCROWN_HOME=/path/to/alpha-beta-CROWN ./setup.sh`.

torch/torchvision are pre-installed as CPU wheels (`torch==2.11.0`) pinned to what
abcrown expects; on a GPU host set `TORCH_INDEX_URL=` to a CUDA wheel index (e.g.
`https://download.pytorch.org/whl/cu124`).

One verifier still relies on an external, non-free piece that a script cannot
install; setup only detects it and prints guidance:

- `milp_abcrown` needs a licensed **CPLEX**. Point setup at a CPLEX Studio
  install with `CPLEX_HOME=/path/to/CPLEX_StudioXXXX ./setup.sh`; setup links its
  full-edition `cplex` CLI onto `PATH`, which `benchmarks.run_pyomo` drives via
  Pyomo's file backend. **Do not** rely on `pip install cplex` — the PyPI wheel
  is the size-capped **Community Edition** (max 1000 vars/constraints, fails with
  `CPLEX Error 1016` on these models). (`milp_abcrown` also uses abcrown bound
  tightening, which setup installs.)

`prune-experiment/recreate.py` builds one command per (method, arch, mode, rate)
cell and runs the ones whose result CSV is missing. It does **not** probe backend
availability — a cell whose verifier is missing (e.g. `milp_abcrown` on a node
without CPLEX) simply fails and is logged, rather than being silently skipped. On
a mixed fleet, keep unavailable methods out of a node's run via `skip.conf`.
Useful setup flags: `./setup.sh --no-torch` (skip torch/abcrown, i.e.
reludiff/neurodiff only) and `./setup.sh --skip-system` (don't touch apt).

### CPLEX on a shared volume (across cluster instances)

CPLEX is the one dependency that cannot be rebuilt from source, so it is the
only thing worth keeping on a persistent volume shared across nodes. Everything
else — the venv, torch, abcrown/`auto_LiRPA`, OpenBLAS, and the ReluDiff/
NeuroDiff binaries — is intentionally rebuilt from source on each node; caching
those would mask reproducibility failures.

Copy the `cplex/` subtree of a CPLEX Studio install onto the volume once (that
subtree, ~120 MB, is all `milp_abcrown` needs — skip `opl/`, `cpoptimizer/`,
etc.; a 1 GB volume is plenty):

```bash
SD=/mnt/exp-data
mkdir -p "$SD/ibm/CPLEX_Studio222"
cp -a /path/to/CPLEX_Studio222/cplex "$SD/ibm/CPLEX_Studio222/"
```

Then on every node that mounts the volume, point setup at it:

```bash
export RUNTIME_DIR="$PWD/runtime"
CPLEX_HOME=/mnt/exp-data/ibm/CPLEX_Studio222 ./setup.sh
```

`setup.sh` resolves `$CPLEX_HOME/cplex/bin/*/cplex`, links that full-edition CLI
onto `PATH`, and `milp_abcrown` solves through it. Note `CPLEX_HOME` is the
directory that *contains* `cplex/`, not the `cplex/` subtree itself.

### Manual setup

Use **Python 3.11** from the repository root (see above for why).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.11.0 torchvision==0.26.0
python3 -m pip install -r requirements.txt
```

Install alpha-beta-CROWN (`abcrown`) and `auto_LiRPA` into the same venv so that
`from abcrown import ABCrownSolver` and `from auto_LiRPA import BoundedModule`
work. `auto_LiRPA` is a git submodule of alpha-beta-CROWN and must be installed
separately from the `abcrown` package (the wheel does not bundle it):

```bash
git clone --recurse-submodules \
  https://github.com/Verified-Intelligence/alpha-beta-CROWN.git "$RUNTIME_DIR/third_party/alpha-beta-CROWN"
python3 -m pip install "$RUNTIME_DIR/third_party/alpha-beta-CROWN/auto_LiRPA"
python3 -m pip install "$RUNTIME_DIR/third_party/alpha-beta-CROWN"
```

The CROWN runner uses abcrown's high-level Python API and writes per-instance
configs/results under `$RUNTIME_DIR/artifacts/abcrown_instances/`. `auto_LiRPA` provides the
ReLU pre-activation bound tightening used by `--bound-tightening abcrown`.

Other external solver/runtime requirements:

- CPLEX: `benchmarks.run_pyomo` solves the MILP encoding with CPLEX, so
  `milp_abcrown` needs a licensed CPLEX Studio install. The backend is chosen by
  `CPLEX_BACKEND`:
  - `auto` (default) — uses the **file** backend when a `cplex` CLI is
    resolvable (from `CPLEX_EXECUTABLE`, `$CPLEX_HOME/cplex/bin/*/cplex`, or
    `PATH`), else the Python API. The CLI shipped in a Studio install is
    full-edition and needs no Python-version-matched bindings.
  - `file` — force the LP/MPS file interface driving the `cplex` CLI.
  - `direct` — force the `cplex_direct` Python API (required for `--debug`
    presolve/progress stats; needs the **full-edition** `cplex` Python bindings,
    not the size-capped PyPI Community Edition wheel).

Download the original ReluDiff MNIST networks and the paper's 100 test inputs:

```bash
python3 scripts/download_mnist_reludiff_nnets.py
```

The downloader reads the files from `DiffNN-Code/nnet` in the official
ReluDiff artifact and validates the architectures before installing them under
`runtime/data/reludiff_mnist/`. In particular, `mnist_relu_3_100` must be
`784-100-100-100-10`; files with architecture `784-100-100-10-10` are rejected.

Check the installed architecture headers without downloading anything:

```bash
python3 scripts/download_mnist_reludiff_nnets.py --check-only
```

The checker reads the files from `runtime/data/reludiff_mnist/` by default. Use
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
  --debug-out runtime/artifacts/cplex_debug.json \
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
  --solver-log-dir runtime/artifacts/solver_logs/sample_highs
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