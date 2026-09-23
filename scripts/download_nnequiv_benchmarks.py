#!/usr/bin/env python3
"""Install the GPE paper's MNIST 8x8 benchmark pairs and input regions.

Fetches the ONNX network pairs from ``samysweb/nnequiv-experiments`` and the
literal input-region bounds from ``samysweb/nnequiv``'s
``examples/equiv/properties.py`` (both at pinned commits, see
``nn_equivalence/nnequiv_benchmarks.py``), converts the networks to this repo's
``.nnet`` format, and installs everything under
``$RUNTIME_DIR/data/nnequiv_mnist8/``::

    regions.json                 every MNIST 8x8 region: explicit low/high vectors
    <pair_id>/net1.nnet          network 1 (the teacher)
    <pair_id>/net2.nnet          network 2 (the "-mirror" student)
    <pair_id>/metadata.json      architectures, the paper's region + property

Every architecture is checked against the paper's published table before
install, so a changed upstream file fails here rather than producing results
for a model nobody trained.

Usage::

    python3 scripts/download_nnequiv_benchmarks.py
    python3 scripts/download_nnequiv_benchmarks.py --force
    python3 scripts/download_nnequiv_benchmarks.py --check-only
    python3 scripts/download_nnequiv_benchmarks.py --stats

``--check-only`` validates what is installed without downloading anything.
``--stats`` additionally reports, per pair over the region centers, the L-inf
logit gap percentiles and the top-1 agreement rate -- the numbers that make an
epsilon ladder defensible, and the first sign that a conversion went wrong.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
import urllib.request
from pathlib import Path

# Make the repo root importable when run as a plain script (python3 scripts/...).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nn_equivalence.nn_types import NeuralNetwork  # noqa: E402
from nn_equivalence.nnequiv_benchmarks import (  # noqa: E402
    EXPERIMENTS_COMMIT,
    EXPERIMENTS_REPO,
    INPUT_DIM,
    MIRROR_SUFFIX,
    NNEQUIV_COMMIT,
    NNEQUIV_REPO,
    PAIRS,
    PIXEL_MAX,
    PIXEL_MIN,
    PROPERTIES_PATH,
    PairSpec,
    is_mnist8x8_region,
    region_center,
    region_radius,
)
from nn_equivalence.onnx_mlp import load_onnx_mlp  # noqa: E402
from nn_equivalence.paths import runtime_path  # noqa: E402
from nn_equivalence.reludiff_nnet import (  # noqa: E402
    load_nnet_layers,
    network_architecture,
    write_nnet_from_scratch,
)

REGIONS_FILE = "regions.json"


def raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch(url: str) -> bytes:
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https)
        return response.read()


# --------------------------------------------------------------------------- #
# Input regions
# --------------------------------------------------------------------------- #


def parse_properties(source: str) -> dict[str, list[list[float]]]:
    """Extract ``PROPERTY["<id>"] = [[uppers], [lowers]]`` literals via AST.

    The upstream file is parsed, never executed: it is downloaded code, and the
    entries we want are plain literals anyway. Assignments built by calling a
    helper (a few non-MNIST ones are) are skipped.
    """
    regions: dict[str, list[list[float]]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            not isinstance(target, ast.Subscript)
            or not isinstance(target.value, ast.Name)
            or target.value.id != "PROPERTY"
        ):
            continue
        try:
            key = ast.literal_eval(target.slice)
            value = ast.literal_eval(node.value)
        except ValueError:
            continue  # e.g. PROPERTY["1000.1"] = scaleInput(...)
        if isinstance(key, str) and isinstance(value, list) and len(value) == 2:
            regions[key] = [[float(x) for x in value[0]], [float(x) for x in value[1]]]
    return regions


def mnist8x8_regions(properties: dict[str, list[list[float]]]) -> dict[str, dict]:
    """Keep the 64-dimensional MNIST 8x8 regions, as explicit low/high vectors."""
    regions: dict[str, dict] = {}
    for key, (uppers, lowers) in properties.items():
        if not is_mnist8x8_region(key) or len(uppers) != INPUT_DIM:
            continue
        for low, high in zip(lowers, uppers):
            if low > high:
                raise ValueError(f"region {key}: lower bound {low} exceeds upper {high}")
            if low < PIXEL_MIN or high > PIXEL_MAX:
                raise ValueError(
                    f"region {key}: bounds [{low}, {high}] leave the "
                    f"[{PIXEL_MIN}, {PIXEL_MAX}] pixel scale"
                )
        regions[key] = {
            "low": lowers,
            "high": uppers,
            "center_id": region_center(key),
            "radius": region_radius(key),
        }
    return regions


def load_regions(output_dir: Path) -> dict[str, dict]:
    path = output_dir / REGIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; install it with "
            "`python3 scripts/download_nnequiv_benchmarks.py`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #


def validate_architecture(spec: PairSpec, role: str, network: NeuralNetwork) -> None:
    expected = spec.net1_arch if role == "net1" else spec.net2_arch
    actual = network_architecture(network)
    if actual != expected:
        raise ValueError(
            f"{spec.pair_id}/{role}: architecture {actual} does not match the "
            f"paper's {expected} for {spec.paper_name}"
        )


def install_pair(spec: PairSpec, output_dir: Path, force: bool) -> None:
    pair_dir = output_dir / spec.pair_id
    metadata_path = pair_dir / "metadata.json"
    if metadata_path.exists() and not force:
        print(f"skip (exists) {pair_dir}  [use --force to overwrite]")
        return

    print(f"{spec.pair_id} ({spec.paper_name})")
    networks: dict[str, NeuralNetwork] = {}
    pair_dir.mkdir(parents=True, exist_ok=True)
    for role, suffix in (("net1", ""), ("net2", MIRROR_SUFFIX)):
        url = raw_url(
            EXPERIMENTS_REPO,
            EXPERIMENTS_COMMIT,
            f"{spec.source_dir}/{spec.source_stem}{suffix}.onnx",
        )
        onnx_path = pair_dir / f"{role}.onnx"
        onnx_path.write_bytes(fetch(url))
        network = load_onnx_mlp(onnx_path)
        validate_architecture(spec, role, network)
        networks[role] = network
        # The regions live on the sklearn-digits 0..16 pixel scale; record that
        # in the .nnet header rather than the writer's [0, 1] default.
        write_nnet_from_scratch(
            network,
            pair_dir / f"{role}.nnet",
            input_mins=[PIXEL_MIN] * INPUT_DIM,
            input_maxes=[PIXEL_MAX] * INPUT_DIM,
        )
        # The .onnx is kept alongside the .nnet (~1 MB for the whole set) so the
        # conversion can be re-checked against onnxruntime without re-downloading.
        print(f"  installed {pair_dir / f'{role}.nnet'}  {network_architecture(network)}")

    metadata_path.write_text(
        json.dumps(
            {
                "pair_id": spec.pair_id,
                "paper_name": spec.paper_name,
                "source": {
                    "repo": EXPERIMENTS_REPO,
                    "commit": EXPERIMENTS_COMMIT,
                    "net1": f"{spec.source_dir}/{spec.source_stem}.onnx",
                    "net2": f"{spec.source_dir}/{spec.source_stem}{MIRROR_SUFFIX}.onnx",
                },
                "net1_arch": network_architecture(networks["net1"]),
                "net2_arch": network_architecture(networks["net2"]),
                "same_architecture": spec.net1_arch == spec.net2_arch,
                "paper_region": spec.paper_region,
                "paper_radius": spec.paper_radius,
                "paper_property": spec.paper_property,
                "paper_epsilon": spec.paper_epsilon,
                "in_table_1": spec.in_table_1,
                "description": spec.description,
                "pixel_scale": [PIXEL_MIN, PIXEL_MAX],
                "training": "student-teacher (knowledge distillation)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  installed {metadata_path}")


# --------------------------------------------------------------------------- #
# Validation and statistics
# --------------------------------------------------------------------------- #


def check_only(output_dir: Path) -> int:
    """Validate installed fixtures. Return a process exit code (0 == all good)."""
    ok = True
    try:
        regions = load_regions(output_dir)
        print(f"OK   {output_dir / REGIONS_FILE} ({len(regions)} regions)")
    except (FileNotFoundError, ValueError) as error:
        ok = False
        regions = {}
        print(f"FAIL {output_dir / REGIONS_FILE}: {error}", file=sys.stderr)

    for spec in PAIRS.values():
        pair_dir = output_dir / spec.pair_id
        try:
            for role in ("net1", "net2"):
                validate_architecture(spec, role, load_nnet_layers(pair_dir / f"{role}.nnet"))
            json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
            if regions and spec.paper_region not in regions:
                raise ValueError(f"paper region {spec.paper_region} missing from regions.json")
            print(f"OK   {pair_dir}")
        except (FileNotFoundError, ValueError, KeyError) as error:
            ok = False
            print(f"FAIL {pair_dir}: {error}", file=sys.stderr)

    if ok:
        print(f"\nAll NNEquiv MNIST 8x8 fixtures present and valid in {output_dir}")
        return 0
    print(f"\nNNEquiv MNIST 8x8 fixtures in {output_dir} are missing or invalid", file=sys.stderr)
    return 1


def _forward(network: NeuralNetwork, inputs: list[float]) -> list[float]:
    from nn_equivalence.encoder_pyomo import forward_values

    return forward_values(network, inputs)


def report_stats(output_dir: Path) -> int:
    """Per pair: L-inf logit gap over the region centers, and top-1 agreement."""
    regions = load_regions(output_dir)
    # One representative point per cluster center: the center of its smallest box.
    centers: dict[str, list[float]] = {}
    for key, region in sorted(regions.items(), key=lambda item: region_radius(item[0])):
        centers.setdefault(
            region["center_id"],
            [(low + high) / 2.0 for low, high in zip(region["low"], region["high"])],
        )
    points = [centers[key] for key in sorted(centers)]
    print(f"statistics over {len(points)} cluster centers\n")

    header = f"{'pair':<24} {'linf gap: min':>13} {'median':>8} {'p95':>8} {'max':>8} {'top-1 agree':>12}"
    print(header)
    print("-" * len(header))
    for spec in PAIRS.values():
        pair_dir = output_dir / spec.pair_id
        net1 = load_nnet_layers(pair_dir / "net1.nnet")
        net2 = load_nnet_layers(pair_dir / "net2.nnet")
        gaps: list[float] = []
        agreements = 0
        for point in points:
            out1, out2 = _forward(net1, point), _forward(net2, point)
            gaps.append(max(abs(a - b) for a, b in zip(out1, out2)))
            agreements += int(out1.index(max(out1)) == out2.index(max(out2)))
        gaps.sort()
        p95 = gaps[min(len(gaps) - 1, int(0.95 * (len(gaps) - 1) + 0.5))]
        print(
            f"{spec.pair_id:<24} {gaps[0]:>13.3f} {statistics.median(gaps):>8.3f} "
            f"{p95:>8.3f} {gaps[-1]:>8.3f} {agreements}/{len(points):>10}"
        )
    print(
        "\nThe paper verifies top-1 equivalence for the -top pairs and "
        "eps-equivalence at eps=15 for mnist_large_epsilon."
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def install(output_dir: Path, force: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    regions_path = output_dir / REGIONS_FILE
    if regions_path.exists() and not force:
        print(f"skip (exists) {regions_path}  [use --force to overwrite]")
    else:
        print("input regions")
        source = fetch(
            raw_url(NNEQUIV_REPO, NNEQUIV_COMMIT, PROPERTIES_PATH)
        ).decode("utf-8")
        regions = mnist8x8_regions(parse_properties(source))
        if not regions:
            raise ValueError(
                f"no MNIST 8x8 regions found in {PROPERTIES_PATH}; upstream layout changed?"
            )
        regions_path.write_text(json.dumps(regions, indent=1) + "\n", encoding="utf-8")
        print(f"  installed {regions_path} ({len(regions)} regions)")

    for spec in PAIRS.values():
        install_pair(spec, output_dir, force)

    print("\nvalidating installed fixtures...")
    return check_only(output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to install / check the fixtures "
        "(default: $RUNTIME_DIR/data/nnequiv_mnist8)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate installed files without downloading anything",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="report per-pair logit-gap and top-1 agreement over the cluster centers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite fixtures that already exist",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Fill the runtime-relative default now (fails fast if RUNTIME_DIR is unset).
    output_dir = args.output_dir or runtime_path("data/nnequiv_mnist8")
    if args.check_only:
        code = check_only(output_dir)
    else:
        code = install(output_dir, args.force)
    if code == 0 and args.stats:
        print()
        code = report_stats(output_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
