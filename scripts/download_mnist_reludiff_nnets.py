#!/usr/bin/env python3
"""Install the ReluDiff MNIST networks and 100-image test set for the benchmarks.

The benchmark suites (`benchmarks/mnist_reludiff.py`, `benchmarks/distillation.py`)
read their fixtures from ``data/reludiff_mnist/``:

  * ``mnist_relu_2_512.nnet``   (784-512-512-10)
  * ``mnist_relu_3_100.nnet``   (784-100-100-10-10)
  * ``mnist_relu_4_1024.nnet``  (784-1024-1024-1024-10-10)
  * ``mnist_tests.h``           (the paper's 100 MNIST images, labels, and the
                                 3 random pixel ids used by the three_pixel mode)

All four files ship inside the official NeuroDiff ASE-2020 artifact under
``DiffNN-Code/`` (``.nnet`` files under ``DiffNN-Code/nnet/``). This script copies
them out of a local checkout of that artifact and validates every architecture
before installing, exactly as the README describes. If the artifact checkout is
missing it is cloned automatically (shallow) unless ``--no-clone`` is given.

Usage::

    python3 scripts/download_mnist_reludiff_nnets.py            # install into data/reludiff_mnist/
    python3 scripts/download_mnist_reludiff_nnets.py --force    # overwrite existing files
    python3 scripts/download_mnist_reludiff_nnets.py --check-only
    python3 scripts/download_mnist_reludiff_nnets.py --output-dir /path --check-only

``--check-only`` validates the installed files (in ``--output-dir``, default
``data/reludiff_mnist``) without downloading anything and exits non-zero if a
file is missing, malformed, or has the wrong architecture.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Make the repo root importable when run as a plain script (python3 scripts/...).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nn_equivalence.reludiff_nnet import (  # noqa: E402
    MNIST_RELUDIFF_NETWORKS,
    load_nnet_layers,
    load_reludiff_mnist_tests,
    validate_mnist_reludiff_network,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "reludiff_mnist"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "third_party" / "NeuroDiff-ASE2020-Artifact"
ARTIFACT_REPO = "https://github.com/pauls658/NeuroDiff-ASE2020-Artifact"

MNIST_TESTS_HEADER = "mnist_tests.h"


def _nnet_names() -> list[str]:
    return [f"{name}.nnet" for name in MNIST_RELUDIFF_NETWORKS]


def _validate_nnet(path: Path) -> None:
    """Raise ValueError/FileNotFoundError if ``path`` is not the expected network."""
    network_name = path.stem
    network = load_nnet_layers(path)
    validate_mnist_reludiff_network(network_name, network, source_path=path)


def _validate_tests(path: Path) -> None:
    """Raise ValueError if the MNIST test header is missing or malformed."""
    load_reludiff_mnist_tests(path)


def check_only(output_dir: Path) -> int:
    """Validate installed files. Return process exit code (0 == all good)."""
    ok = True
    for filename in _nnet_names():
        path = output_dir / filename
        try:
            _validate_nnet(path)
            print(f"OK   {path}")
        except (FileNotFoundError, ValueError) as error:
            ok = False
            print(f"FAIL {path}: {error}", file=sys.stderr)

    tests_path = output_dir / MNIST_TESTS_HEADER
    try:
        _validate_tests(tests_path)
        print(f"OK   {tests_path}")
    except (FileNotFoundError, ValueError) as error:
        ok = False
        print(f"FAIL {tests_path}: {error}", file=sys.stderr)

    if ok:
        print(f"\nAll ReluDiff MNIST fixtures present and valid in {output_dir}")
        return 0
    print(f"\nReluDiff MNIST fixtures in {output_dir} are missing or invalid", file=sys.stderr)
    return 1


def ensure_artifact(artifact_dir: Path, allow_clone: bool) -> Path:
    """Return the DiffNN-Code directory inside the NeuroDiff artifact, cloning it
    into ``artifact_dir`` if necessary."""
    diffnn = artifact_dir / "DiffNN-Code"
    if diffnn.is_dir():
        return diffnn

    if not allow_clone:
        raise FileNotFoundError(
            f"NeuroDiff artifact not found at {artifact_dir} (and --no-clone was "
            f"given). Clone it first:\n  git clone --depth 1 {ARTIFACT_REPO} {artifact_dir}"
        )

    print(f"NeuroDiff artifact not found at {artifact_dir}; cloning (shallow)...")
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", ARTIFACT_REPO, str(artifact_dir)],
        check=True,
    )
    if not diffnn.is_dir():
        raise FileNotFoundError(
            f"cloned {ARTIFACT_REPO} but {diffnn} is missing; artifact layout changed?"
        )
    return diffnn


def install(output_dir: Path, artifact_dir: Path, force: bool, allow_clone: bool) -> int:
    """Copy + validate the fixtures into ``output_dir``. Return an exit code."""
    diffnn = ensure_artifact(artifact_dir, allow_clone)
    nnet_src_dir = diffnn / "nnet"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Map each destination file to its source inside the artifact.
    sources: list[tuple[Path, Path]] = [
        (nnet_src_dir / filename, output_dir / filename) for filename in _nnet_names()
    ]
    sources.append((diffnn / MNIST_TESTS_HEADER, output_dir / MNIST_TESTS_HEADER))

    missing_src = [src for src, _ in sources if not src.exists()]
    if missing_src:
        print(
            "these files are missing from the NeuroDiff artifact:\n  "
            + "\n  ".join(str(path) for path in missing_src),
            file=sys.stderr,
        )
        return 1

    for src, dst in sources:
        if dst.exists() and not force:
            print(f"skip (exists) {dst}  [use --force to overwrite]")
            continue
        shutil.copyfile(src, dst)
        print(f"installed {dst}")

    # Validate what we just installed so a bad copy fails loudly here, not deep
    # inside a benchmark run.
    print("\nvalidating installed fixtures...")
    return check_only(output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where to install / check the fixtures (default: data/reludiff_mnist)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="local NeuroDiff ASE-2020 artifact checkout "
        "(default: third_party/NeuroDiff-ASE2020-Artifact)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate installed files without downloading anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite fixtures that already exist",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="fail instead of cloning the artifact when it is missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        return check_only(args.output_dir)
    return install(
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        force=args.force,
        allow_clone=not args.no_clone,
    )


if __name__ == "__main__":
    raise SystemExit(main())
