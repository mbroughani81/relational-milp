"""The GPE paper's MNIST 8x8 equivalence benchmarks.

Teuber et al., "Geometric Path Enumeration for Equivalence Verification of
Neural Networks" (2021) publish their benchmark networks and input regions in
two repositories:

  * ``samysweb/nnequiv-experiments`` — the ONNX network pairs (``<stem>.onnx``
    is network 1, ``<stem>-mirror.onnx`` is network 2) plus ``instances.csv``
    listing which (network, input region, property) triples were run;
  * ``samysweb/nnequiv`` — the tool, whose ``examples/equiv/properties.py``
    holds the literal per-dimension input bounds behind each region id.

This module is the single place that records that mapping. Both
``scripts/download_nnequiv_benchmarks.py`` (which installs the pairs) and
``benchmarks/suites/distillation_mnist8.py`` (which verifies them) import it.

Region ids are the paper's: the integer hundred is the cluster center
(``9000`` .. ``9900``, ten centers from the Kleine Büning et al. MNIST 8x8
set) and the remainder is the L-infinity radius on the sklearn-digits 0..16
pixel scale, so ``9000.7`` is center ``9000`` at radius ``0.7`` and ``9103``
is center ``9100`` at radius ``3``. Bounds are clamped to ``[0, 16]``, which
is why the published boxes are not always symmetric about the center.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pinned so a rerun installs byte-identical fixtures.
EXPERIMENTS_REPO = "samysweb/nnequiv-experiments"
EXPERIMENTS_COMMIT = "dba603534f1ead46449a2118e11cb4c634b18525"
NNEQUIV_REPO = "samysweb/nnequiv"
NNEQUIV_COMMIT = "ffc42dbd08a11277dac65b55db731d2ffb66bad9"

# Where properties.py lives inside the tool repo.
PROPERTIES_PATH = "examples/equiv/properties.py"

# sklearn "digits" pixel scale; the published regions are clamped to it.
PIXEL_MIN = 0.0
PIXEL_MAX = 16.0
INPUT_DIM = 64

MIRROR_SUFFIX = "-mirror"


@dataclass(frozen=True)
class PairSpec:
    """One benchmark pair: two networks plus the region/property the paper used."""

    pair_id: str
    paper_name: str
    source_dir: str  # repo subdirectory holding <source_stem>{,-mirror}.onnx
    source_stem: str
    net1_arch: list[int]
    net2_arch: list[int]
    paper_region: str  # region id, e.g. "9000.7"
    paper_property: str  # "linf" or "top1"
    paper_epsilon: float | None  # only for paper_property == "linf"
    in_table_1: bool  # appears in the paper's Table I / Fig. 3 comparison
    description: str

    @property
    def paper_radius(self) -> float:
        return region_radius(self.paper_region)


# The five Table I benchmarks come from instances-versions.csv, which is
# authoritative: benchmarks.md duplicates the "MNIST_large-epsilon" row and
# labels the KL-loss pairs "-epsilon" even though they are run as top-1.
PAIRS: dict[str, PairSpec] = {
    "mnist_small_top": PairSpec(
        pair_id="mnist_small_top",
        paper_name="MNIST_small-top",
        source_dir="benchmarks-versions",
        source_stem="mnist8x8_student_36_10",
        net1_arch=[64, 32, 16, 10],
        net2_arch=[64, 36, 10],
        paper_region="9000.7",
        paper_property="top1",
        paper_epsilon=None,
        in_table_1=True,
        description="Student-teacher; shallower, wider student (32-16 -> 36)",
    ),
    "mnist_medium_top": PairSpec(
        pair_id="mnist_medium_top",
        paper_name="MNIST_medium-top",
        source_dir="benchmarks-versions",
        source_stem="mnist8x8_student_12_12_12_10",
        net1_arch=[64, 32, 16, 10],
        net2_arch=[64, 12, 12, 12, 10],
        paper_region="9000.7",
        paper_property="top1",
        paper_epsilon=None,
        in_table_1=True,
        description="Student-teacher; deeper, narrower student (32-16 -> 12-12-12)",
    ),
    "mnist_large_top": PairSpec(
        pair_id="mnist_large_top",
        paper_name="MNIST_large-top",
        source_dir="benchmarks-versions",
        source_stem="mnist8x8_100_80_60_40_20_10",
        net1_arch=[64, 100, 80, 60, 40, 20, 10],
        net2_arch=[64, 40, 40, 10],
        paper_region="9200.2",
        paper_property="top1",
        paper_epsilon=None,
        in_table_1=True,
        description="Student-teacher with KL-divergence loss; 6-layer teacher",
    ),
    "mnist_larger_top": PairSpec(
        pair_id="mnist_larger_top",
        paper_name="MNIST_larger-top",
        source_dir="benchmarks-versions",
        source_stem="mnist8x8_100_200_300_200_100_10",
        net1_arch=[64, 100, 200, 300, 200, 100, 10],
        net2_arch=[64, 40, 40, 10],
        paper_region="9200.2",
        paper_property="top1",
        paper_epsilon=None,
        in_table_1=True,
        description="Student-teacher with KL-divergence loss; widest teacher",
    ),
    "mnist_large_epsilon": PairSpec(
        pair_id="mnist_large_epsilon",
        paper_name="MNIST_large-epsilon",
        source_dir="benchmarks-versions",
        source_stem="mnist8x8_100_80_60_40_20_10_eps1",
        net1_arch=[64, 100, 80, 60, 40, 20, 10],
        net2_arch=[64, 40, 40, 10],
        paper_region="9000.2",
        paper_property="linf",
        paper_epsilon=15.0,
        in_table_1=True,
        description="Student-teacher with MSE loss; the paper's only MNIST eps run",
    ),
    "mnist_student_24_12_10": PairSpec(
        pair_id="mnist_student_24_12_10",
        paper_name="mnist8x8_student_24_12_10",
        # Only shipped under benchmarks/; absent from benchmarks-versions/.
        source_dir="benchmarks",
        source_stem="mnist8x8_student_24_12_10",
        net1_arch=[64, 32, 16, 10],
        net2_arch=[64, 24, 12, 10],
        paper_region="9000.5",
        paper_property="top1",
        paper_epsilon=None,
        in_table_1=False,
        description="Third student of the same teacher; in instances.csv, not Table I",
    ),
}

# Regions the paper exercises for MNIST 8x8: ten cluster centers, and the radii
# published for them. Not every (center, radius) pair exists upstream - the
# downloader installs whatever properties.py actually defines.
REGION_CENTERS: tuple[str, ...] = tuple(f"9{index}00" for index in range(10))
REGION_RADII: tuple[float, ...] = (0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 3.0)


def region_center(region_id: str) -> str:
    """Cluster-center id behind a region id (``"9000.7"`` -> ``"9000"``)."""
    return str(int(float(region_id) // 100) * 100)


def region_radius(region_id: str) -> float:
    """L-infinity radius behind a region id (``"9000.7"`` -> ``0.7``).

    Rounded: subtracting the center leaves binary-float noise (``9200.2`` less
    ``9200`` is ``0.2000000000007``), and the radius is round-tripped through
    :func:`region_id` to look regions up again.
    """
    return round(float(region_id) - int(float(region_id) // 100) * 100, 6)


def region_id(center: str, radius: float) -> str:
    """Region id for a center and radius; the inverse of the two accessors above.

    Upstream keys are the plain decimal rendering of ``center + radius``, so
    radius ``1.0`` on center ``9000`` is ``"9001"``, not ``"9000.1"``.
    """
    value = float(center) + float(radius)
    text = f"{value:.10g}"
    return text


def is_mnist8x8_region(region_id_: str) -> bool:
    """True for the MNIST 8x8 region ids (the 9000-9999 block)."""
    try:
        value = float(region_id_)
    except ValueError:
        return False
    return 9000.0 <= value < 10000.0


def paper_pair_ids() -> tuple[str, ...]:
    """Pair ids that appear in the paper's Table I / Fig. 3 comparison."""
    return tuple(
        pair_id for pair_id, spec in PAIRS.items() if spec.in_table_1
    )
