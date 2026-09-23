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

There are two sweeps, each a `<name>-experiment/recreate.py` that builds one
command per grid cell and runs the ones whose result CSV is missing:

| sweep | grid | verifiers |
|---|---|---|
| `prune-experiment` | (method, arch, mode, prune rate) over `pruning_mnist` | all four |
| `distillation-experiment` | (pair, property, epsilon *or* radius) over `distillation_mnist8` | `milp_abcrown` only |

Neither **probes backend availability** — a cell whose verifier is missing
(e.g. `milp_abcrown` on a node without CPLEX) simply fails and is logged,
rather than being silently skipped. On a mixed fleet, keep unavailable methods
out of a node's run via that sweep's `skip.conf`. Useful setup flags:
`./setup.sh --no-torch` (skip torch/abcrown, i.e. reludiff/neurodiff only) and
`./setup.sh --skip-system` (don't touch apt).

`scripts/worker.py` works either sweep's queue across a fleet, claiming cells
with an atomic `mkdir` lock on the shared filesystem. `EXPERIMENT` selects
which (default `prune`):

```bash
EXPERIMENT=distillation scripts/worker.py              # work the queue
EXPERIMENT=distillation PROGRESS=1 scripts/worker.py   # show progress, exit
```

Each experiment gets its own shared-FS subdirectory (override with `RUN`), so
the two sweeps never share a queue. Keep `RUN` stable across passes: the run
directory is what lets a later pass resume unfinished cells rather than redo
them.

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

Download the GPE paper's MNIST 8x8 teacher/student pairs and input regions
(needed by the `distillation_mnist8` suite):

```bash
python3 scripts/download_nnequiv_benchmarks.py
```

This fetches the ONNX network pairs from `samysweb/nnequiv-experiments` and the
literal input-region bounds from `samysweb/nnequiv`'s
`examples/equiv/properties.py` (both at commits pinned in
`nn_equivalence/nnequiv_benchmarks.py`), validates every architecture against
the paper's published table, and installs `.nnet` conversions under
`runtime/data/nnequiv_mnist8/`. `--check-only` re-validates without
downloading; `--stats` additionally prints each pair's L-inf logit gap and
top-1 agreement over the ten cluster centers.

## Run benchmarks

The verifiers share the same `--suite` / repeated `--suite-options KEY=VALUE`
interface. The suites are `pruning_mnist`, `distillation_mnist` and
`distillation_mnist8` (described under
[Current benchmark suites](#current-benchmark-suites)).

Relational MILP with CPLEX (interval ReLU bounds):

```bash
python3 -m benchmarks.run_pyomo \
  --suite pruning_mnist \
  --solver cplex \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global,three_pixel \
  --suite-options limit=3 \
  --suite-options timeout=10
```

Tighten the ReLU pre-activation bounds with alpha-beta-CROWN before solving
(`--bound-tightening interval|abcrown`):

```bash
python3 -m benchmarks.run_pyomo \
  --suite pruning_mnist \
  --solver cplex \
  --bound-tightening abcrown \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
  --suite-options limit=3
```

Run alpha-beta-CROWN on its own (`--profile` selects a named config):

```bash
python3 -m benchmarks.run_crown \
  --suite pruning_mnist \
  --profile relu-kfsb \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
  --suite-options limit=3
```

Run ReluDiff / NeuroDiff (the `delta_network_test` C verifiers) on the same
`pruning_mnist` instances as the MILP and CROWN runners:

```bash
python3 -m benchmarks.run_diffverifier \
  --tool neurodiff \
  --binary /path/to/delta_network_test \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
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

Write the CSV results to a file with `--csv` (all runners support it):

```bash
python3 -m benchmarks.run_pyomo \
  --suite pruning_mnist \
  --solver cplex \
  --suite-options networks=mnist_relu_3_100 \
  --suite-options modes=global \
  --suite-options limit=3 \
  --csv results.csv
```

### CPLEX debug and presolve statistics

Capture structured CPLEX statistics without running a second diagnostic
presolve pass:

```bash
python3 -m benchmarks.run_pyomo \
  --suite pruning_mnist \
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

## Current benchmark suites

- `pruning_mnist`: compares the original ReluDiff `.nnet` MNIST models with a
  magnitude-pruned copy of the *same* architecture. For image label `c`, each
  instance verifies `|nn1(x)[c] - nn2(x)[c]| <= epsilon`. Supports `networks`,
  `modes`, `limit`, `timeout`, `epsilon`, `radius`, and `sparsity` suite options.
- `distillation_mnist`: Hinton-style knowledge-distillation teacher/student
  pairs. Property A is numerical logit equivalence on the labeled class:
  `|z_T(x)[c] - z_S(x)[c]| <= epsilon`. Uses the same ReluDiff 100-image /
  3-pixel fixtures as `pruning_mnist`. Supports `pairs`, `tiers`, `modes`,
  `limit`, `timeout`, `epsilon`, and `radius`.

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
  --suite distillation_mnist \
  --solver cplex \
  --suite-options tiers=A \
  --suite-options modes=three_pixel \
  --suite-options epsilon=2.0 \
  --suite-options limit=100 \
  --suite-options timeout=30
```

Run ReluDiff / NeuroDiff on Tier A only:

```bash
python3 -m benchmarks.run_diffverifier \
  --suite distillation_mnist \
  --tool neurodiff \
  --binary /path/to/delta_network_test \
  --suite-options pairs=kd_a1 \
  --suite-options modes=three_pixel \
  --suite-options epsilon=2.0 \
  --suite-options limit=100 \
  --suite-options timeout=30
```

- `distillation_mnist8`: the **GPE paper's own** MNIST 8x8 knowledge-distillation
  pairs (Teuber et al. 2021), verified under the paper's two properties so our
  solve times sit next to their published NNEquiv / MilpEquiv numbers
  (Table I, Fig. 3-4). Networks are 64-input (8x8 digits on the sklearn 0..16
  pixel scale); install them with `scripts/download_nnequiv_benchmarks.py`.
  Supports `pairs`, `property`, `epsilon`, `radius`, `centers`, `limit`,
  `timeout`, `data_dir`.

  | property | meaning | `epsilon` means |
  |---|---|---|
  | `linf` | `max_i \|z1(x)_i - z2(x)_i\| <= epsilon` (their Def. 1) | the bound; default `15.0`, the paper's value |
  | `top1` | `argmax z1(x) == argmax z2(x)` (their Def. 2) | a **strictness margin**, not an output tolerance; default `1e-4` |

  The `top1` margin exists because a counterexample is a *strict* violation and
  a MILP cannot state a strict inequality. It must stay above the solver's
  feasibility tolerance (see `encoder_pyomo.TOP1_MIN_MARGIN`): at `0` every tie
  point would be reported as a counterexample.

  Every pair differs in architecture, so **ReluDiff / NeuroDiff cannot run this
  suite**, and `run_crown` only encodes a single-output margin so it cannot
  express either property — the suite is `run_pyomo` only.

  Each pair is verified at the ten published cluster centers. The one
  (pair, region, property) combination the paper reports proving carries
  `expected_status="unsat"`, so a replication failure shows up in the results
  CSV as `expected=unsat:no`. Each instance also records `center_linf_gap` and
  `center_top1_agrees` — how the two networks already differ at the region's
  center point — so the analysis can tell a genuine proof from a benchmark that
  was never going to hold.

```bash
python3 -m benchmarks.run_pyomo \
  --suite distillation_mnist8 \
  --solver cplex \
  --bound-tightening abcrown \
  --suite-options pairs=mnist_small_top \
  --suite-options property=top1 \
  --suite-options limit=10 \
  --suite-options timeout=1200
```

Other historically documented suites (`sample`, `synthetic`, `bigger_synthetic`,
`mnist`) may not be present in the current tree.