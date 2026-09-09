from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common import Hyperrectangle, Instance, validate_instance
from benchmarks.distillation import load_suite
from nn_equivalence.reludiff_nnet import (
    load_nnet_layers,
    network_architecture,
    write_nnet_from_scratch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "distillation" / "mnist"
TESTS_PATH = REPO_ROOT / "data" / "reludiff_mnist" / "mnist_tests.h"
PAIR = "kd_1"

requires_pair = pytest.mark.skipif(
    not (DATA_DIR / PAIR / "teacher.nnet").exists()
    or not (DATA_DIR / PAIR / "student.nnet").exists()
    or not TESTS_PATH.exists(),
    reason="distillation kd_1 pair or ReluDiff tests missing",
)


def _tiny_network(widths: list[int]) -> list:
    layers = []
    for input_size, output_size in zip(widths, widths[1:]):
        weights = [[0.01 * (i + j + 1) for j in range(input_size)] for i in range(output_size)]
        bias = [0.1 * (i + 1) for i in range(output_size)]
        layers.append((weights, bias))
    return layers


def test_validate_instance_allows_unequal_hidden_widths() -> None:
    instance = Instance(
        instance_id="unequal",
        suite_name="test",
        nn1=_tiny_network([2, 4, 3]),
        nn2=_tiny_network([2, 5, 3]),
        input_region=Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0]),
        epsilon=0.1,
        output_index=1,
    )
    validate_instance(instance)


def test_validate_instance_allows_unequal_depth() -> None:
    instance = Instance(
        instance_id="unequal_depth",
        suite_name="test",
        nn1=_tiny_network([2, 4, 3]),
        nn2=_tiny_network([2, 3]),
        input_region=Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0]),
        epsilon=0.1,
        output_index=0,
    )
    validate_instance(instance)


def test_validate_instance_rejects_mismatched_output_size() -> None:
    instance = Instance(
        instance_id="bad_out",
        suite_name="test",
        nn1=_tiny_network([2, 3]),
        nn2=_tiny_network([2, 4]),
        input_region=Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0]),
        epsilon=0.1,
    )
    with pytest.raises(ValueError, match="same output size"):
        validate_instance(instance)


def test_write_nnet_from_scratch_roundtrip(tmp_path: Path) -> None:
    network = _tiny_network([4, 3, 2])
    path = tmp_path / "toy.nnet"
    write_nnet_from_scratch(network, path)
    loaded = load_nnet_layers(path)
    assert network_architecture(loaded) == [4, 3, 2]
    assert loaded == network


def test_distillation_suite_loads_from_fixture_pair(tmp_path: Path, monkeypatch) -> None:
    if not TESTS_PATH.exists():
        pytest.skip("ReluDiff tests missing")

    pair_dir = tmp_path / "kd_toy"
    pair_dir.mkdir()
    teacher = _tiny_network([784, 8, 10])
    student = _tiny_network([784, 4, 10])
    write_nnet_from_scratch(teacher, pair_dir / "teacher.nnet")
    write_nnet_from_scratch(student, pair_dir / "student.nnet")
    (pair_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pair_id": "kd_toy",
                "tier": "B",
                "teacher_arch": [784, 8, 10],
                "student_arch": [784, 4, 10],
                "same_architecture": False,
                "temperature": 2.0,
                "alpha": 0.5,
                "seed": 0,
                "teacher_accuracy": 0.9,
                "student_accuracy": 0.85,
                "disagreement_rate": 0.05,
                "teacher_param_count": 100,
                "student_param_count": 50,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(REPO_ROOT)
    suite = load_suite(
        {
            "pairs": "kd_toy",
            "modes": "three_pixel",
            "limit": "2",
            "epsilon": "0.25",
            "timeout": "7",
            "data_dir": str(tmp_path),
            "tests_path": str(TESTS_PATH),
        }
    )
    assert suite.name == "distillation"
    assert len(suite.instances) == 2
    for sample_index, instance in enumerate(suite.instances):
        assert instance.instance_id == f"kd_toy_three_pixel_{sample_index}"
        assert instance.epsilon == 0.25
        assert instance.timeout_sec == 7.0
        assert network_architecture(instance.nn1) == [784, 8, 10]
        assert network_architecture(instance.nn2) == [784, 4, 10]
        assert instance.metadata["pair_id"] == "kd_toy"
        assert instance.metadata["property"] == "output_equivalence_logits"
        validate_instance(instance)


@requires_pair
def test_load_kd_1_suite(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    suite = load_suite(
        {
            "pairs": PAIR,
            "modes": "three_pixel",
            "limit": "3",
            "epsilon": "0.1",
        }
    )
    assert len(suite.instances) == 3
    for instance in suite.instances:
        assert network_architecture(instance.nn1) == [784, 64, 32, 10]
        assert network_architecture(instance.nn2) == [784, 32, 16, 10]
        assert instance.metadata["tier"] == "B"
        assert instance.metadata["same_architecture"] == 0
        validate_instance(instance)


@requires_pair
def test_tiers_filter_selects_matching_pairs(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    suite = load_suite(
        {
            "tiers": "B",
            "modes": "three_pixel",
            "limit": "1",
            "epsilon": "0.1",
        }
    )
    assert suite.instances
    assert all(inst.metadata["tier"] == "B" for inst in suite.instances)
