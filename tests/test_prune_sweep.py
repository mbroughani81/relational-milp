"""Tests for the prune-experiment sweep runner (prune-experiment/recreate.py)."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RECREATE_PATH = REPO_ROOT / "prune-experiment" / "recreate.py"


def _load_recreate():
    spec = importlib.util.spec_from_file_location("prune_recreate", RECREATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recreate = _load_recreate()


def test_grid_cell_count():
    cells = list(recreate.iter_cells())
    expected = (
        len(recreate.METHODS)
        * len(recreate.ARCHS)
        * len(recreate.MODES)
        * len(recreate.RATES)
    )
    assert len(cells) == expected
    # Tags are unique.
    tags = [tag for *_, tag in cells]
    assert len(set(tags)) == len(tags)


def test_plan_objects_have_result_and_command():
    plan = recreate.build_plan(skip_patterns=[])
    assert len(plan) == len(list(recreate.iter_cells()))
    for obj in plan:
        assert set(obj.keys()) == {"result", "command"}
        assert obj["result"].endswith(".csv")
        assert "-m benchmarks.run_" in obj["command"]


def test_skip_pattern_excludes_matching_tags():
    all_plan = recreate.build_plan(skip_patterns=[])
    # Skip every milp_abcrown cell.
    filtered = recreate.build_plan(skip_patterns=["milp_abcrown__*"])
    assert len(filtered) < len(all_plan)
    assert not any("milp_abcrown" in obj["result"] for obj in filtered)
    # Non-milp cells survive.
    assert any("reludiff" in obj["result"] for obj in filtered)


@pytest.mark.parametrize(
    "method,needle",
    [
        ("milp_abcrown", "run_pyomo"),
        ("abcrown", "run_crown"),
        ("reludiff", "run_diffverifier"),
        ("neurodiff", "run_diffverifier"),
    ],
)
def test_build_command_uses_expected_runner(method, needle):
    out = REPO_ROOT / "prune-experiment" / "results" / "x.csv"
    cmd = recreate.build_command(method, "mnist_relu_3_100", "global", 10, out)
    assert any(needle in part for part in cmd)
    assert cmd[-2:] == ["--csv", str(out)]
    if method in ("reludiff", "neurodiff"):
        assert "--tool" in cmd and method in cmd
    if method == "milp_abcrown":
        assert "--solver" in cmd and "cplex" in cmd
