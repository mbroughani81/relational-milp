"""Phase 3 guards for the unified runner, result schema, and CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from nnequiv.core import (
    Hyperrectangle,
    Instance,
    InstanceResult,
    InstanceSuite,
    SolveStats,
)
from nnequiv.report.schema import CORE_COLUMNS, build_rows, rows_to_csv
from nnequiv.run.selection import extract_selection, select_instances


def _instance(instance_id: str, property_kind: str = "logit_class") -> Instance:
    net = [([[1.0]], [0.0])]
    return Instance(
        instance_id=instance_id,
        suite_name="s",
        nn1=net,
        nn2=net,
        input_region=Hyperrectangle(low=[0.0], high=[1.0]),
        epsilon=1.0,
        property_kind=property_kind,
    )


def _result(instance_id: str, status: str = "unsat", stats=None) -> InstanceResult:
    return InstanceResult(
        instance_id=instance_id,
        suite_name="s",
        status=status,
        runtime_sec=0.5,
        epsilon=1.0,
        expected_status=None,
        stats=stats or [],
    )


# --- selection -------------------------------------------------------------


def test_extract_selection_pops_limit():
    remaining, limit, ids = extract_selection({"networks": "x", "limit": "3"})
    assert remaining == {"networks": "x"}
    assert limit == 3
    assert ids == []


def test_extract_selection_parses_ids():
    remaining, limit, ids = extract_selection({"ids": "a, b ,c"})
    assert remaining == {}
    assert limit is None
    assert ids == ["a", "b", "c"]


def test_extract_selection_rejects_both():
    with pytest.raises(ValueError, match="cannot be combined"):
        extract_selection({"limit": "1", "ids": "a"})


def test_select_instances_limit_and_ids():
    instances = [_instance("a"), _instance("b"), _instance("c")]
    assert [i.instance_id for i in select_instances(instances, 2, [])] == ["a", "b"]
    assert [i.instance_id for i in select_instances(instances, None, ["c"])] == ["c"]


def test_select_instances_unknown_id_raises():
    with pytest.raises(ValueError, match="unknown instance ids"):
        select_instances([_instance("a")], None, ["z"])


def test_select_instances_requires_limit_or_ids():
    with pytest.raises(ValueError, match="either suite option"):
        select_instances([_instance("a")], None, [])


# --- report schema ---------------------------------------------------------


def test_report_core_columns_and_property():
    rows = build_rows("milp", [_instance("a", "top1")], [_result("a", "sat")])
    lines = rows_to_csv(rows).splitlines()
    header = lines[0].split(",")
    assert header[: len(CORE_COLUMNS)] == list(CORE_COLUMNS)
    fields = lines[1].split(",")
    assert fields[0] == "a"  # instance_id
    assert fields[2] == "milp"  # verifier
    assert fields[3] == "top1"  # property (from the instance)
    assert fields[4] == "sat"  # status


def test_report_extra_columns_namespaced_and_unioned():
    results = [
        _result(
            "a",
            stats=[SolveStats(name="diff", details=[("diff_status", "no_adv"), ("num_splits", 7)])],
        ),
        _result(
            "b",
            stats=[SolveStats(name="diff", details=[("diff_status", "adv_found")])],
        ),
    ]
    rows = build_rows("diff", [_instance("a"), _instance("b")], results)
    lines = rows_to_csv(rows).splitlines()
    header = lines[0].split(",")
    assert "extra.diff.diff_status" in header
    assert "extra.diff.num_splits" in header

    splits_index = header.index("extra.diff.num_splits")
    assert lines[1].split(",")[splits_index] == "7"
    # row b never reported num_splits -> blank under the unioned column.
    assert lines[2].split(",")[splits_index] == ""


def test_report_expected_and_matched():
    result = InstanceResult(
        instance_id="a",
        suite_name="s",
        status="unsat",
        runtime_sec=0.1,
        epsilon=1.0,
        expected_status="unsat",
    )
    lines = rows_to_csv(build_rows("milp", [_instance("a")], [result])).splitlines()
    header = lines[0].split(",")
    fields = lines[1].split(",")
    assert fields[header.index("expected")] == "unsat"
    assert fields[header.index("matched")] == "yes"


# --- runner ----------------------------------------------------------------


class _FakeVerifier:
    supported_properties = frozenset({"logit_class"})

    def __init__(self):
        self.seen = None

    def supports(self, property_kind):
        return property_kind in self.supported_properties

    def verify_suite(self, instances):
        self.seen = list(instances)
        return [_result(instance.instance_id) for instance in instances]


def test_run_suite_selects_then_delegates(monkeypatch):
    import nnequiv.run.runner as runner

    instances = [_instance("a"), _instance("b"), _instance("c")]
    monkeypatch.setattr(
        runner, "load_suite", lambda name, options: InstanceSuite(name=name, instances=instances)
    )
    fake = _FakeVerifier()
    monkeypatch.setattr(runner, "create", lambda name, options: fake)

    selected, results = runner.run_suite("fake", "s", {"limit": "2"}, {})
    assert [i.instance_id for i in selected] == ["a", "b"]
    assert [r.instance_id for r in results] == ["a", "b"]
    assert [i.instance_id for i in fake.seen] == ["a", "b"]


def test_run_suite_capability_guard(monkeypatch):
    import nnequiv.run.runner as runner

    instances = [_instance("a", "top1")]
    monkeypatch.setattr(
        runner, "load_suite", lambda name, options: InstanceSuite(name=name, instances=instances)
    )

    class CrownLike:
        supported_properties = frozenset({"logit_class"})

        def supports(self, property_kind):
            return property_kind in self.supported_properties

        def verify_suite(self, instances):
            raise AssertionError("verify_suite must not run when the guard trips")

    monkeypatch.setattr(runner, "create", lambda name, options: CrownLike())
    with pytest.raises(ValueError, match="cannot express property"):
        runner.run_suite("crown", "s", {"limit": "1"}, {})


# --- CLI -------------------------------------------------------------------


def test_cli_run_help():
    result = subprocess.run(
        [sys.executable, "-m", "nnequiv.run.cli", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--verifier" in result.stdout


def test_cli_parses_options_and_prints_csv(monkeypatch, capsys):
    import nnequiv.run.cli as cli

    called = {}

    def fake_run_suite(verifier, suite, suite_options, verifier_options):
        called.update(
            verifier=verifier,
            suite=suite,
            suite_options=suite_options,
            verifier_options=verifier_options,
        )
        return [_instance("a")], [_result("a")]

    monkeypatch.setattr(cli, "run_suite", fake_run_suite)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "run",
                "--verifier",
                "milp",
                "--suite",
                "pruning_mnist",
                "-o",
                "limit=1",
                "--verifier-opt",
                "bound_tightening=abcrown",
            ]
        )
    assert exit_info.value.code == 0
    assert called["verifier"] == "milp"
    assert called["suite"] == "pruning_mnist"
    assert called["suite_options"] == {"limit": "1"}
    assert called["verifier_options"] == {"bound_tightening": "abcrown"}
    assert "instance_id,suite,verifier,property,status" in capsys.readouterr().out
