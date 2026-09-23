"""The GPE paper's MNIST 8x8 knowledge-distillation equivalence benchmarks.

Verifies the student-teacher network pairs of Teuber et al. 2021 ("Geometric
Path Enumeration for Equivalence Verification of Neural Networks") over the
paper's own input regions, under the paper's two properties:

  ``linf``  max_i |z1(x)_i - z2(x)_i| <= epsilon   (their Definition 1)
  ``top1``  argmax z1(x) == argmax z2(x)           (their Definition 2)

Fixtures come from ``scripts/download_nnequiv_benchmarks.py``, which installs
them under ``$RUNTIME_DIR/data/nnequiv_mnist8/``. Networks are 64-input (8x8
digits on the sklearn 0..16 pixel scale); every pair differs in architecture,
so ReluDiff/NeuroDiff cannot run these and the suite is MILP-only.

Each pair is verified at ten cluster centers. The paper picks one specific
(pair, center, radius, property) per benchmark -- that instance carries
``expected_status="unsat"``, since the paper reports proving it, which makes
a replication failure visible in the results CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from benchmarks.common import (
    AbstractPolytope,
    EquivalenceProperty,
    Hyperrectangle,
    Instance,
    InstanceSuite,
    SuiteOptions,
)
from nn_equivalence.nn_types import NeuralNetwork
from nn_equivalence.nnequiv_benchmarks import PAIRS, PairSpec, region_id
from nn_equivalence.paths import runtime_path
from nn_equivalence.reludiff_nnet import load_nnet_layers, network_architecture

SUPPORTED_PROPERTIES: tuple[EquivalenceProperty, ...] = ("linf", "top1")

# epsilon means different things per property, so its default does too:
#   linf  the paper's only MNIST epsilon run uses epsilon=15
#   top1  a strictness margin, not a tolerance on the outputs; see
#         encoder_pyomo.TOP1_MIN_MARGIN for why it cannot be 0
DEFAULT_EPSILON = {"linf": "15.0", "top1": "1e-4"}

REGIONS_FILE = "regions.json"

DEFAULT_SUITE_OPTIONS: SuiteOptions = {
    "pairs": "",
    "property": "linf",
    # Empty -> resolved from the property / each pair's paper setting below.
    "epsilon": "",
    "radius": "",
    "centers": "",
    "timeout": "600",
    "data_dir": "",
}

ALLOWED_OPTIONS = frozenset(DEFAULT_SUITE_OPTIONS) | {"limit"}


def _normalized_options(suite_options: SuiteOptions | None) -> SuiteOptions:
    options: SuiteOptions = dict(DEFAULT_SUITE_OPTIONS)
    for key, value in (suite_options or {}).items():
        options[key.strip().lower().replace("-", "_")] = value

    unknown_options = set(options) - ALLOWED_OPTIONS
    if unknown_options:
        raise ValueError(
            f"unknown distillation_mnist8 suite options: {sorted(unknown_options)}"
        )
    return options


def _option_tuple(options: SuiteOptions, name: str) -> tuple[str, ...]:
    value = options.get(name)
    if not value:
        return tuple()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _property(options: SuiteOptions) -> EquivalenceProperty:
    value = options["property"].strip()
    if value not in SUPPORTED_PROPERTIES:
        raise ValueError(
            f"unknown distillation_mnist8 property {value!r}; "
            f"expected one of {list(SUPPORTED_PROPERTIES)}"
        )
    return value  # type: ignore[return-value]


def _epsilon(options: SuiteOptions, property_kind: EquivalenceProperty) -> float:
    value = options["epsilon"].strip() or DEFAULT_EPSILON[property_kind]
    epsilon = float(value)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    return epsilon


def _limit(options: SuiteOptions) -> int | None:
    value = options.get("limit")
    if not value:
        return None
    limit = int(value)
    if limit < 1:
        raise ValueError("distillation_mnist8 limit must be at least 1")
    return limit


def _resolve_pair_ids(options: SuiteOptions) -> tuple[str, ...]:
    pair_ids = _option_tuple(options, "pairs")
    if not pair_ids:
        return tuple(PAIRS)
    unknown_pairs = [pair_id for pair_id in pair_ids if pair_id not in PAIRS]
    if unknown_pairs:
        raise ValueError(
            f"unknown distillation_mnist8 pairs: {unknown_pairs}; "
            f"available: {sorted(PAIRS)}"
        )
    return pair_ids


def load_regions(data_dir: Path) -> dict[str, dict]:
    path = data_dir / REGIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"missing input regions at {path}; install the fixtures with "
            "`python3 scripts/download_nnequiv_benchmarks.py`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_centers(
    options: SuiteOptions,
    regions: dict[str, dict],
) -> tuple[str, ...]:
    available = sorted({region["center_id"] for region in regions.values()})
    centers = _option_tuple(options, "centers")
    if not centers:
        return tuple(available)
    unknown_centers = [center for center in centers if center not in available]
    if unknown_centers:
        raise ValueError(
            f"unknown distillation_mnist8 centers: {unknown_centers}; "
            f"available: {available}"
        )
    return centers


def _load_pair(data_dir: Path, spec: PairSpec) -> tuple[NeuralNetwork, NeuralNetwork]:
    pair_dir = data_dir / spec.pair_id
    paths = [pair_dir / "net1.nnet", pair_dir / "net2.nnet"]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing network files: "
            + ", ".join(str(path) for path in missing)
            + ". Install them with "
            "`python3 scripts/download_nnequiv_benchmarks.py`."
        )
    return load_nnet_layers(paths[0]), load_nnet_layers(paths[1])


def _region_polytope(region: dict) -> AbstractPolytope:
    return Hyperrectangle(low=list(region["low"]), high=list(region["high"]))


def _center_point(region: dict) -> list[float]:
    return [(low + high) / 2.0 for low, high in zip(region["low"], region["high"])]


def _center_behavior(
    nn1: NeuralNetwork,
    nn2: NeuralNetwork,
    point: list[float],
) -> tuple[float, bool]:
    """How the two networks already differ at a region's center point.

    Recorded per instance so the analysis can tell a benchmark the property
    genuinely holds on from one where the networks visibly disagree before
    any search starts.
    """
    from nn_equivalence.encoder_pyomo import forward_values

    out1 = forward_values(nn1, point)
    out2 = forward_values(nn2, point)
    linf_gap = max(abs(a - b) for a, b in zip(out1, out2))
    agrees = out1.index(max(out1)) == out2.index(max(out2))
    return linf_gap, agrees


def load_suite(suite_options: SuiteOptions | None = None) -> InstanceSuite:
    suite_name = "distillation_mnist8"
    options = _normalized_options(suite_options)
    print(f"{suite_name} suite options: {options}", file=sys.stderr)

    data_dir = (
        Path(options["data_dir"])
        if options["data_dir"]
        else runtime_path("data/nnequiv_mnist8")
    )
    regions = load_regions(data_dir)
    pair_ids = _resolve_pair_ids(options)
    centers = _resolve_centers(options, regions)
    property_kind = _property(options)
    epsilon = _epsilon(options, property_kind)
    timeout_sec = float(options["timeout"])
    limit = _limit(options)
    # Empty radius means "each pair's own radius from the paper".
    radius_override = options["radius"].strip()

    instances: list[Instance] = []
    for pair_id in pair_ids:
        spec = PAIRS[pair_id]
        nn1, nn2 = _load_pair(data_dir, spec)
        radius = float(radius_override) if radius_override else spec.paper_radius
        pair_instances = 0
        for center in centers:
            key = region_id(center, radius)
            if key not in regions:
                raise ValueError(
                    f"no published input region for center {center} at radius "
                    f"{radius:g} (looked for {key!r}); available radii for this "
                    f"center: {sorted({r['radius'] for r in regions.values() if r['center_id'] == center})}"
                )
            if limit is not None and pair_instances >= limit:
                break
            region = regions[key]
            linf_gap, top1_agrees = _center_behavior(nn1, nn2, _center_point(region))

            # The paper reports proving exactly this configuration, so treat a
            # different outcome as a replication failure rather than a result.
            is_paper_cell = (
                property_kind == spec.paper_property
                and key == spec.paper_region
                and (
                    property_kind == "top1"
                    or spec.paper_epsilon is None
                    or epsilon == spec.paper_epsilon
                )
            )
            instances.append(
                Instance(
                    instance_id=f"{pair_id}_{property_kind}_{key}",
                    suite_name=suite_name,
                    nn1=nn1,
                    nn2=nn2,
                    input_region=_region_polytope(region),
                    epsilon=epsilon,
                    property_kind=property_kind,
                    expected_status="unsat" if is_paper_cell else None,
                    timeout_sec=timeout_sec,
                    metadata={
                        "pair_id": pair_id,
                        "paper_name": spec.paper_name,
                        "property": property_kind,
                        "region_id": key,
                        "center_id": center,
                        "radius": radius,
                        "net1_arch": "-".join(
                            str(size) for size in network_architecture(nn1)
                        ),
                        "net2_arch": "-".join(
                            str(size) for size in network_architecture(nn2)
                        ),
                        "in_table_1": int(spec.in_table_1),
                        "is_paper_cell": int(is_paper_cell),
                        "center_linf_gap": linf_gap,
                        "center_top1_agrees": int(top1_agrees),
                    },
                )
            )
            pair_instances += 1

    return InstanceSuite(name=suite_name, instances=instances)
