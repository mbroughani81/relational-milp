import struct
from pathlib import Path

import pytest

from benchmarks.mnist_reludiff import load_suite
from nn_equivalence.paths import runtime_path
from nn_equivalence.reludiff_nnet import (
    MNIST_RELUDIFF_ARCHITECTURES,
    network_architecture,
    prune_network_unstructured,
    quantize_network_float16,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = runtime_path("data/reludiff_mnist")
# The 3x100 network is the smallest bundled .nnet, so it loads quickly.
NETWORK = "mnist_relu_3_100"

requires_data = pytest.mark.skipif(
    not (DATA_DIR / f"{NETWORK}.nnet").exists()
    or not (DATA_DIR / "mnist_tests.h").exists(),
    reason=(
        "ReluDiff MNIST data not present under runtime/data/reludiff_mnist; "
        "download the .nnet fixtures to run this test"
    ),
)


def _float16_round(value: float) -> float:
    """Round a Python float to the nearest IEEE-754 half-precision value."""
    return float(struct.unpack("e", struct.pack("e", float(value)))[0])


def _weight_count(network) -> int:
    return sum(len(row) for weights, _ in network for row in weights)


def _weight_zeros(network) -> int:
    return sum(
        1
        for weights, _ in network
        for row in weights
        for value in row
        if value == 0.0
    )


@pytest.fixture
def suite(monkeypatch):
    # load_suite reads $RUNTIME_DIR/data/reludiff_mnist (set by tests/conftest.py).
    monkeypatch.chdir(REPO_ROOT)
    return load_suite(
        {
            "networks": NETWORK,
            "modes": "global",
            "limit": "3",
        }
    )


@pytest.fixture
def pruned_suite(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    return load_suite(
        {
            "networks": NETWORK,
            "modes": "global",
            "limit": "3",
            "perturbation": "prune",
            "sparsity": "0.5",
        }
    )


@requires_data
def test_load_suite_builds_expected_instances(suite):
    # limit=3, one network, one mode => three instances.
    assert len(suite.instances) == 3
    for sample_index, instance in enumerate(suite.instances):
        assert instance.suite_name == "mnist_reludiff"
        assert instance.instance_id == f"{NETWORK}_global_{sample_index}"


@requires_data
def test_load_suite_networks_follow_expected_architecture(suite):
    expected = MNIST_RELUDIFF_ARCHITECTURES[NETWORK]
    for instance in suite.instances:
        # Both the original and the quantized network must keep the exact
        # ReluDiff benchmark architecture.
        assert network_architecture(instance.nn1) == expected
        assert network_architecture(instance.nn2) == expected


@requires_data
def test_load_suite_quantizes_second_network_to_float16(suite):
    for instance in suite.instances:
        # nn2 is exactly the float16 quantization of nn1.
        assert instance.nn2 == quantize_network_float16(instance.nn1)

        # Quantization actually changed something: a trained float32 network
        # is not already representable in half precision.
        assert instance.nn1 != instance.nn2

        # Every nn2 value is a genuine float16 value: re-quantizing is a no-op.
        assert quantize_network_float16(instance.nn2) == instance.nn2


def test_quantize_network_float16_rounds_to_half_precision():
    # 0.1 is not representable in float16; it rounds to the nearest half value.
    original: list = [([[0.1, -0.3]], [1.0])]
    quantized = quantize_network_float16(original)

    (weights, bias) = quantized[0]
    assert weights[0][0] == pytest.approx(0.0999755859375, abs=0.0)
    assert weights[0][0] != 0.1
    # 1.0 and -0.3? -0.3 also shifts; 1.0 stays exact.
    assert bias[0] == 1.0

    # Shape is preserved and the result is float16-stable (idempotent).
    assert network_architecture(quantized) == network_architecture(original)
    assert quantize_network_float16(quantized) == quantized

    # Values match a direct half-precision round-trip.
    assert weights[0][1] == _float16_round(-0.3)


def test_prune_network_unstructured_zeros_exact_count_globally():
    # 6 weights total; magnitudes: 4, 3, 2, 1, 0.5, 0.25.
    network = [
        ([[3.0, -1.0], [0.5, 2.0]], [10.0, 20.0]),
        ([[-4.0, 0.25]], [30.0]),
    ]
    pruned = prune_network_unstructured(network, 0.5)

    # floor(0.5 * 6) == 3 smallest weights zeroed: 0.25, 0.5, 1.0.
    assert _weight_zeros(pruned) == 3
    assert pruned == [
        ([[3.0, 0.0], [0.0, 2.0]], [10.0, 20.0]),
        ([[-4.0, 0.0]], [30.0]),
    ]

    # Architecture preserved; biases untouched; the 3 largest weights survive.
    assert network_architecture(pruned) == network_architecture(network)
    assert [bias for _, bias in pruned] == [[10.0, 20.0], [30.0]]

    # Does not mutate the input network.
    assert _weight_zeros(network) == 0


def test_prune_network_unstructured_zero_sparsity_is_noop():
    network = [([[0.1, -0.3]], [1.0])]
    pruned = prune_network_unstructured(network, 0.0)
    assert pruned == network
    assert pruned is not network  # returns a fresh copy


@pytest.mark.parametrize("bad_sparsity", [-0.1, 1.0, 2.0])
def test_prune_network_unstructured_rejects_out_of_range(bad_sparsity):
    with pytest.raises(ValueError):
        prune_network_unstructured([([[1.0]], [0.0])], bad_sparsity)


@requires_data
def test_load_suite_prune_keeps_architecture(pruned_suite):
    expected = MNIST_RELUDIFF_ARCHITECTURES[NETWORK]
    for instance in pruned_suite.instances:
        # Unstructured pruning does not change shape: both nets keep the arch.
        assert network_architecture(instance.nn1) == expected
        assert network_architecture(instance.nn2) == expected


@requires_data
def test_load_suite_prune_applies_exact_sparsity(pruned_suite):
    for instance in pruned_suite.instances:
        # nn2 is exactly the pruned version of nn1.
        assert instance.nn2 == prune_network_unstructured(instance.nn1, 0.5)

        # Realized sparsity is exactly floor(0.5 * total_weights) zeros. The
        # original network has no exact-zero weights, so every zero is pruned.
        total = _weight_count(instance.nn2)
        assert _weight_zeros(instance.nn2) == int(0.5 * total)

        # Pruning genuinely changed the network.
        assert instance.nn1 != instance.nn2


@requires_data
def test_load_suite_prune_sets_metadata(pruned_suite):
    for instance in pruned_suite.instances:
        assert instance.metadata["perturbation"] == "prune"
        assert instance.metadata["sparsity"] == 0.5
        assert "quantization" not in instance.metadata
