"""Tests for the distillation sweep runner (distillation-experiment/recreate.py)."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RECREATE_PATH = REPO_ROOT / "distillation-experiment" / "recreate.py"


def _load_recreate():
    spec = importlib.util.spec_from_file_location("distillation_recreate", RECREATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recreate = _load_recreate()


def test_grid_cell_count():
    cells = list(recreate.iter_cells())
    # linf sweeps epsilon, top1 sweeps radius; both over every pair.
    expected = len(recreate.METHODS) * len(recreate.PAIR_IDS) * (
        len(recreate.EPSILONS) + len(recreate.RADII)
    )
    assert len(cells) == expected
    tags = [tag for *_, tag in cells]
    assert len(set(tags)) == len(tags)


def test_top1_cells_sweep_radius_and_have_no_epsilon_axis():
    top1_cells = [cell for cell in recreate.iter_cells() if cell[2] == "top1"]
    knobs_per_pair = {}
    for _method, pair_id, _property, knob, _tag in top1_cells:
        knobs_per_pair.setdefault(pair_id, set()).add(knob)
    assert all(knobs == set(recreate.RADII) for knobs in knobs_per_pair.values())

    for _method, pair_id, _property, knob, _tag in top1_cells:
        command = recreate.build_command(
            "milp_abcrown", pair_id, "top1", knob, Path("x.csv")
        )
        assert f"radius={knob:g}" in command
        # epsilon is present only as the strictness margin.
        assert f"epsilon={recreate.TOP1_MARGIN}" in command


def test_linf_cells_sweep_epsilon_at_each_pairs_published_radius():
    linf_cells = [cell for cell in recreate.iter_cells() if cell[2] == "linf"]
    for _method, pair_id, _property, knob, _tag in linf_cells:
        command = recreate.build_command(
            "milp_abcrown", pair_id, "linf", knob, Path("x.csv")
        )
        assert f"epsilon={knob:g}" in command
        # No radius option -> the suite falls back to the paper's per-pair one.
        assert not any(part.startswith("radius=") for part in command)


def test_the_papers_epsilon_is_in_the_ladder():
    """Keeps one cell directly comparable with the paper's Table I."""
    assert 15.0 in recreate.EPSILONS


def test_plan_objects_have_result_and_command():
    plan = recreate.build_plan(skip_patterns=[])
    assert len(plan) == len(list(recreate.iter_cells()))
    for obj in plan:
        assert set(obj.keys()) == {"result", "command"}
        assert obj["result"].endswith(".csv")
        assert "-m benchmarks.run_pyomo" in obj["command"]
        assert "--suite distillation_mnist8" in obj["command"]


def test_skip_pattern_excludes_matching_tags():
    all_plan = recreate.build_plan(skip_patterns=[])
    filtered = recreate.build_plan(skip_patterns=["*__top1__*"])
    assert 0 < len(filtered) < len(all_plan)
    assert not any("__top1__" in obj["result"] for obj in filtered)
    assert any("__linf__" in obj["result"] for obj in filtered)


def test_build_command_rejects_verifiers_that_cannot_express_the_properties():
    with pytest.raises(ValueError, match="cannot express"):
        recreate.build_command("abcrown", "mnist_small_top", "linf", 15.0, Path("x.csv"))
    with pytest.raises(ValueError, match="unknown method"):
        recreate.build_command("reludiff", "mnist_small_top", "linf", 15.0, Path("x.csv"))


def test_equal_instance_budget_across_properties():
    # linf runs two separately timed directions, top1 one symmetric solve.
    assert recreate.timeout_for("linf") * 2 == recreate.INSTANCE_BUDGET
    assert recreate.timeout_for("top1") == recreate.INSTANCE_BUDGET
