"""Phase 2 guards for the Verifier seam.

These test the abstraction, the registry, the capability matrix, and that each
adapter wires to the existing runner correctly (via monkeypatch, so no solver /
torch / C binary is needed) -- not the underlying solve correctness, which the
encoder/suite tests and real runs already cover.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nnequiv.core import Hyperrectangle, Instance, InstanceResult
from nnequiv.verifiers import (
    CrownVerifier,
    DiffVerifier,
    MilpVerifier,
    Verifier,
    available,
    create,
)


def _tiny_instance(suite_name: str = "pruning_mnist") -> Instance:
    net = [([[1.0]], [0.0])]
    return Instance(
        instance_id="i",
        suite_name=suite_name,
        nn1=net,
        nn2=net,
        input_region=Hyperrectangle(low=[0.0], high=[1.0]),
        epsilon=1.0,
    )


def _fake_result(instance_id: str = "i") -> InstanceResult:
    return InstanceResult(
        instance_id=instance_id,
        suite_name="s",
        status="unsat",
        runtime_sec=0.0,
        epsilon=1.0,
        expected_status=None,
    )


def test_registry_lists_and_creates_builtins():
    assert available() == ["crown", "diff", "milp"]
    assert isinstance(create("milp"), MilpVerifier)
    assert isinstance(create("crown"), CrownVerifier)
    assert isinstance(create("diff"), DiffVerifier)


def test_create_unknown_verifier_raises():
    with pytest.raises(ValueError, match="unknown verifier 'bogus'"):
        create("bogus")


def test_supports_matrix():
    # Only the MILP verifier can express linf / top1; CROWN and the diff tools
    # do the single-output logit_class property only.
    assert MilpVerifier().supports("logit_class")
    assert MilpVerifier().supports("linf")
    assert MilpVerifier().supports("top1")

    for verifier in (CrownVerifier(), DiffVerifier()):
        assert verifier.supports("logit_class")
        assert not verifier.supports("linf")
        assert not verifier.supports("top1")


def test_from_options_parsing():
    milp = create("milp", {"bound_tightening": "abcrown", "fix_stable_relu_binaries": "no"})
    assert isinstance(milp, MilpVerifier)
    assert milp.bound_tightening == "abcrown"
    assert milp.fix_stable_relu_binaries is False

    crown = create("crown", {"profile": "relu-kfsb"})
    assert isinstance(crown, CrownVerifier)
    assert crown.profile == "relu-kfsb"

    diff = create("diff", {"tool": "reludiff", "binary": "/path/to/delta"})
    assert isinstance(diff, DiffVerifier)
    assert diff.tool == "reludiff"
    assert diff.binary == "/path/to/delta"


def test_verify_delegates_to_verify_suite():
    class FakeVerifier(Verifier):
        name = "fake"
        supported_properties = frozenset({"logit_class"})

        def __init__(self):
            self.seen: list[list[Instance]] = []

        def verify_suite(self, instances):
            self.seen.append(list(instances))
            return [_fake_result(inst.instance_id) for inst in instances]

        @classmethod
        def from_options(cls, options):
            return cls()

    verifier = FakeVerifier()
    instance = _tiny_instance()
    result = verifier.verify(instance)
    assert result.instance_id == "i"
    # verify() must route through verify_suite with exactly one instance.
    assert verifier.seen == [[instance]]


def test_milp_verify_delegates_to_run_instance(monkeypatch):
    import nnequiv.verifiers.milp.runner as milp_runner

    calls = []
    sentinel = _fake_result()

    def fake_run_instance(
        instance, solver_name, bound_tightening, *, verbose, debug, fix_stable_relu_binaries
    ):
        calls.append(
            (instance, solver_name, bound_tightening, verbose, debug, fix_stable_relu_binaries)
        )
        return sentinel

    monkeypatch.setattr(milp_runner, "run_instance", fake_run_instance)

    verifier = MilpVerifier(
        solver="cplex",
        bound_tightening="abcrown",
        fix_stable_relu_binaries=False,
        verbose=True,
        debug=False,
    )
    instance = _tiny_instance()
    result = verifier.verify(instance)

    assert result is sentinel
    assert calls == [(instance, "cplex", "abcrown", True, False, False)]


def test_crown_verify_suite_is_batched(monkeypatch):
    import nnequiv.verifiers.crown.runner as crown_runner

    calls = {"count": 0, "instances": None}

    def fake_prepare(suite, profile):
        return Path("cfg.yaml"), suite.instances

    def fake_run_abcrown(config_path, instances, profile, callback):
        calls["count"] += 1
        calls["instances"] = instances
        return 0, "", {i: ("safe", 0.1) for i in range(len(instances))}

    monkeypatch.setattr(crown_runner, "prepare_artifacts", fake_prepare)
    monkeypatch.setattr(crown_runner, "run_abcrown", fake_run_abcrown)

    instances = [_tiny_instance(), _tiny_instance()]
    results = CrownVerifier(profile="relu-kfsb").verify_suite(instances)

    # CROWN is batch-native: one run_abcrown call for the whole suite, not one
    # per instance. build_results (run for real) maps "safe" -> unsat.
    assert calls["count"] == 1
    assert len(calls["instances"]) == 2
    assert [result.status for result in results] == ["unsat", "unsat"]


def test_importing_verifiers_is_torch_free():
    code = (
        "import sys\n"
        "import nnequiv.verifiers as v\n"
        "assert 'torch' not in sys.modules, 'verifiers import pulled in torch'\n"
        "assert v.available() == ['crown', 'diff', 'milp'], v.available()\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
