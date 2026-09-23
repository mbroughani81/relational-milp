"""Tests for the GPE-paper MNIST 8x8 suite (benchmarks/suites/distillation_mnist8.py)."""

from __future__ import annotations

import pytest

from benchmarks.common import validate_instance
from benchmarks.suites.distillation_mnist8 import load_suite
from nn_equivalence.nnequiv_benchmarks import (
    PAIRS,
    region_center,
    region_id,
    region_radius,
)
from nn_equivalence.paths import runtime_path

DATA_DIR = runtime_path("data/nnequiv_mnist8")
PAIR = "mnist_small_top"

requires_fixtures = pytest.mark.skipif(
    not (DATA_DIR / "regions.json").exists()
    or not (DATA_DIR / PAIR / "net1.nnet").exists(),
    reason=(
        "NNEquiv MNIST 8x8 fixtures missing; install with "
        "`python3 scripts/download_nnequiv_benchmarks.py`"
    ),
)


# --------------------------------------------------------------------------- #
# Region id arithmetic (no fixtures needed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key,center,radius",
    [
        ("9000.7", "9000", 0.7),
        ("9000.01", "9000", 0.01),
        ("9200.2", "9200", 0.2),
        # Radius 1 and 3 render without a decimal point upstream.
        ("9001", "9000", 1.0),
        ("9103", "9100", 3.0),
        ("9900.8", "9900", 0.8),
    ],
)
def test_region_id_round_trips(key, center, radius):
    assert region_center(key) == center
    assert region_radius(key) == radius
    assert region_id(center, radius) == key


# --------------------------------------------------------------------------- #
# Suite options
# --------------------------------------------------------------------------- #


@requires_fixtures
def test_property_defaults_to_linf_with_the_papers_epsilon():
    suite = load_suite({"pairs": PAIR, "centers": "9000"})
    instance = suite.instances[0]
    assert instance.property_kind == "linf"
    assert instance.epsilon == 15.0


@requires_fixtures
def test_top1_gets_a_strictness_margin_not_an_output_tolerance():
    suite = load_suite({"pairs": PAIR, "property": "top1", "centers": "9000"})
    assert suite.instances[0].property_kind == "top1"
    assert suite.instances[0].epsilon == 1e-4


@requires_fixtures
def test_unknown_property_is_rejected():
    with pytest.raises(ValueError, match="unknown distillation_mnist8 property"):
        load_suite({"pairs": PAIR, "property": "logit_class"})


@requires_fixtures
def test_unknown_option_pair_and_center_are_rejected():
    with pytest.raises(ValueError, match="unknown distillation_mnist8 suite options"):
        load_suite({"pairs": PAIR, "perturb": "3"})
    with pytest.raises(ValueError, match="unknown distillation_mnist8 pairs"):
        load_suite({"pairs": "no_such_pair"})
    with pytest.raises(ValueError, match="unknown distillation_mnist8 centers"):
        load_suite({"pairs": PAIR, "centers": "1234"})


@requires_fixtures
def test_radius_without_a_published_region_fails_loudly():
    with pytest.raises(ValueError, match="no published input region"):
        load_suite({"pairs": PAIR, "property": "top1", "radius": "0.42"})


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #


@requires_fixtures
def test_each_pair_gets_one_instance_per_cluster_center():
    suite = load_suite({"pairs": PAIR, "property": "top1"})
    assert len(suite.instances) == 10
    assert len({instance.metadata["center_id"] for instance in suite.instances}) == 10
    for instance in suite.instances:
        validate_instance(instance)


@requires_fixtures
def test_default_radius_is_each_pairs_published_one():
    suite = load_suite({"property": "top1", "centers": "9000"})
    by_pair = {
        instance.metadata["pair_id"]: instance.metadata["radius"]
        for instance in suite.instances
    }
    for pair_id, radius in by_pair.items():
        assert radius == PAIRS[pair_id].paper_radius
    # An explicit radius overrides every pair's default.
    overridden = load_suite({"property": "top1", "centers": "9000", "radius": "0.3"})
    assert {instance.metadata["radius"] for instance in overridden.instances} == {0.3}


@requires_fixtures
def test_input_region_matches_the_published_bounds():
    suite = load_suite({"pairs": PAIR, "property": "top1", "centers": "9000"})
    instance = suite.instances[0]
    bounds = instance.input_region.bounds()
    assert len(bounds) == 64
    assert all(low <= high for low, high in bounds)
    # sklearn digits pixel scale, clamped.
    assert min(low for low, _ in bounds) >= 0.0
    assert max(high for _, high in bounds) <= 16.0


@requires_fixtures
def test_the_papers_own_cell_is_marked_and_expected_to_prove():
    suite = load_suite({"pairs": PAIR, "property": "top1"})
    paper_cells = [
        instance for instance in suite.instances if instance.metadata["is_paper_cell"]
    ]
    assert len(paper_cells) == 1
    instance = paper_cells[0]
    assert instance.metadata["region_id"] == PAIRS[PAIR].paper_region
    # The paper reports proving it, so a different outcome is a replication failure.
    assert instance.expected_status == "unsat"
    assert all(
        other.expected_status is None
        for other in suite.instances
        if not other.metadata["is_paper_cell"]
    )


@requires_fixtures
def test_a_property_the_paper_did_not_run_marks_no_paper_cell():
    # mnist_small_top is a top-1 benchmark; the paper never ran linf on it.
    suite = load_suite({"pairs": PAIR, "property": "linf"})
    assert not any(instance.metadata["is_paper_cell"] for instance in suite.instances)


@requires_fixtures
def test_center_behavior_is_recorded_for_the_analysis():
    suite = load_suite({"pairs": PAIR, "property": "top1"})
    for instance in suite.instances:
        assert instance.metadata["center_linf_gap"] >= 0.0
        assert instance.metadata["center_top1_agrees"] in (0, 1)
