import struct
from pathlib import Path

import pytest

from benchmarks.mnist_reludiff import load_suite
from nn_equivalence.reludiff_nnet import (
    MNIST_RELUDIFF_ARCHITECTURES,
    network_architecture,
    quantize_network_float16,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "reludiff_mnist"
# The 3x100 network is the smallest bundled .nnet, so it loads quickly.
NETWORK = "mnist_relu_3_100"

requires_data = pytest.mark.skipif(
    not (DATA_DIR / f"{NETWORK}.nnet").exists()
    or not (DATA_DIR / "mnist_tests.h").exists(),
    reason=(
        "ReluDiff MNIST data not present under data/reludiff_mnist; "
        "download the .nnet fixtures to run this test"
    ),
)


def _float16_round(value: float) -> float:
    """Round a Python float to the nearest IEEE-754 half-precision value."""
    return float(struct.unpack("e", struct.pack("e", float(value)))[0])


@pytest.fixture
def suite(monkeypatch):
    # load_suite reads data/reludiff_mnist relative to the cwd.
    monkeypatch.chdir(REPO_ROOT)
    return load_suite(
        {
            "networks": NETWORK,
            "modes": "global",
            "limit": "3",
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
