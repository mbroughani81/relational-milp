from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common import Hyperrectangle, Instance
from benchmarks.run_diffverifier import (
    build_command,
    classify_output,
    parse_tool_stats,
    property_id_for,
)
from nn_equivalence.reludiff_nnet import (
    load_nnet_layers,
    prune_network_unstructured,
    write_nnet_layers,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "reludiff_mnist"
NETWORK = "mnist_relu_3_100"

requires_data = pytest.mark.skipif(
    not (DATA_DIR / f"{NETWORK}.nnet").exists(),
    reason="ReluDiff MNIST .nnet fixtures not present under data/reludiff_mnist",
)


def _instance(metadata: dict) -> Instance:
    net = [([[1.0, 2.0]], [0.0])]
    return Instance(
        instance_id="x",
        suite_name="mnist_reludiff",
        nn1=net,
        nn2=net,
        input_region=Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0]),
        epsilon=1.0,
        metadata=metadata,
    )


def test_build_command_global_uses_perturb_flag():
    command = build_command("bin", 400, "a.nnet", "b.nnet", 1.0, "global", 3.0)
    assert command == ["bin", "400", "a.nnet", "b.nnet", "1", "-p", "3"]


def test_build_command_three_pixel_uses_x_flag_and_omits_perturb():
    command = build_command("bin", 442, "a.nnet", "b.nnet", 0.5, "three_pixel", 3.0)
    assert command == ["bin", "442", "a.nnet", "b.nnet", "0.5", "-x", "3"]
    assert "-p" not in command


def test_build_command_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_command("bin", 400, "a", "b", 1.0, "elsewhere", 3.0)


def test_classify_output_verified_is_unsat():
    assert classify_output("running property 400", "\nNo adv!\ntime: 0.1", False) == (
        "unsat",
        "no_adv",
    )


def test_classify_output_adv_is_sat():
    stderr = "\nadv found:\nadv is: ...\ntime: 0.2"
    assert classify_output("", stderr, False) == ("sat", "adv_found")


def test_classify_output_timeout():
    assert classify_output("", "", True) == ("timeout", "timeout")


def test_classify_output_no_conclusion_is_unknown():
    assert classify_output("running property 400", "some noise", False) == (
        "unknown",
        "no_conclusion",
    )


def test_parse_tool_stats_extracts_time_and_splits():
    stats = parse_tool_stats("No adv!\ntime: 2.422256 \n\n\nnumSplits: 2619")
    assert stats == {"tool_time_sec": 2.422256, "num_splits": 2619}


def test_parse_tool_stats_missing_fields_returns_empty():
    assert parse_tool_stats("running property with no verdict") == {}


def test_property_id_maps_sample_index_to_400_range():
    assert property_id_for(_instance({"network": NETWORK, "sample_index": 0})) == 400
    assert property_id_for(_instance({"network": NETWORK, "sample_index": 42})) == 442


def test_property_id_rejects_missing_sample_index():
    with pytest.raises(ValueError):
        property_id_for(_instance({"network": NETWORK}))


def test_property_id_rejects_out_of_range_sample_index():
    with pytest.raises(ValueError):
        property_id_for(_instance({"network": NETWORK, "sample_index": 100}))


@requires_data
def test_write_nnet_layers_round_trips_pruned_network(tmp_path):
    source = DATA_DIR / f"{NETWORK}.nnet"
    original = load_nnet_layers(source)
    pruned = prune_network_unstructured(original, 0.5)

    out = tmp_path / "pruned.nnet"
    write_nnet_layers(source, pruned, out)

    # The C tools re-read the file; our loader is the reference parser. Writing
    # then reloading must reproduce the exact pruned parameters, and the copied
    # header must keep the architecture intact.
    reloaded = load_nnet_layers(out)
    assert reloaded == pruned


@requires_data
def test_write_nnet_layers_rejects_architecture_mismatch(tmp_path):
    source = DATA_DIR / f"{NETWORK}.nnet"
    wrong_shape = [([[1.0, 2.0, 3.0]], [0.0])]
    with pytest.raises(ValueError):
        write_nnet_layers(source, wrong_shape, tmp_path / "bad.nnet")


def test_distillation_nnet_paths_rejects_tier_b(tmp_path):
    from benchmarks.run_diffverifier import _distillation_nnet_paths

    teacher = [([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]), ([[1.0, 0.0]], [0.0])]
    student = [([[1.0, 0.0]], [0.0]), ([[1.0]], [0.0])]
    instance = Instance(
        instance_id="kd_1_three_pixel_0",
        suite_name="distillation",
        nn1=teacher,
        nn2=student,
        input_region=Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0]),
        epsilon=1.0,
        metadata={
            "pair_id": "kd_1",
            "tier": "B",
            "same_architecture": 0,
            "sample_index": 0,
        },
    )
    with pytest.raises(ValueError, match="Tier B"):
        _distillation_nnet_paths(instance, tmp_path, {})


def test_distillation_nnet_paths_writes_same_arch_student(tmp_path):
    from benchmarks.run_diffverifier import _distillation_nnet_paths
    from nn_equivalence.reludiff_nnet import write_nnet_from_scratch

    teacher = [
        ([[0.1, 0.2], [0.3, 0.4]], [0.0, 0.0]),
        ([[0.5, 0.6]], [0.1]),
    ]
    student = [
        ([[0.9, 0.8], [0.7, 0.6]], [0.2, 0.3]),
        ([[0.4, 0.3]], [0.5]),
    ]
    teacher_path = tmp_path / "teacher.nnet"
    write_nnet_from_scratch(teacher, teacher_path)
    instance = Instance(
        instance_id="kd_a1_three_pixel_0",
        suite_name="distillation",
        nn1=teacher,
        nn2=student,
        input_region=Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0]),
        epsilon=1.0,
        metadata={
            "pair_id": "kd_a1",
            "tier": "A",
            "same_architecture": 1,
            "nnet1_path": str(teacher_path),
            "sample_index": 0,
        },
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    nnet1, nnet2 = _distillation_nnet_paths(instance, work_dir, {})
    assert nnet1 == teacher_path
    assert load_nnet_layers(nnet2) == student
