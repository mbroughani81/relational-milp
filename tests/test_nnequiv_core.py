"""Guards for the Phase 1 clean core.

These lock in the three things the core/bounds extraction is supposed to buy:
a single canonical home for the domain model, backward-compatible facades at the
old import sites, and a dependency direction where the pure core (and the MILP
encoder that now depends on it) never pulls in torch or the benchmarks harness.
"""

from __future__ import annotations

import subprocess
import sys

from nnequiv.bounds import network_bounds
from nnequiv.core import Hyperrectangle, Instance, validate_instance


def test_benchmarks_common_reexports_core_identity():
    import benchmarks.common as common
    import nnequiv.core as core

    # The facade must expose the *same* objects, not lookalikes.
    assert common.Instance is core.Instance
    assert common.Hyperrectangle is core.Hyperrectangle
    assert common.validate_instance is core.validate_instance


def test_nn_types_shim_reexports_core_types():
    import nn_equivalence.nn_types as nn_types
    from nnequiv.core import types

    assert nn_types.NeuralNetwork is types.NeuralNetwork
    assert nn_types.Bounds is types.Bounds


def test_network_bounds_interval_matches_hand_computation():
    # y = x0 - x1 over x in [0, 1]^2, so the single output lies in [-1, 1].
    network = [([[1.0, -1.0]], [0.0])]
    bounds = network_bounds(network, [(0.0, 1.0), (0.0, 1.0)], "interval")
    assert bounds == [[(-1.0, 1.0)]]


def test_validate_instance_still_enforced_via_core():
    net = [([[1.0]], [0.0])]
    good = Instance(
        instance_id="x",
        suite_name="s",
        nn1=net,
        nn2=net,
        input_region=Hyperrectangle(low=[0.0], high=[1.0]),
        epsilon=0.0,
    )
    validate_instance(good)  # must not raise


def test_pure_core_and_encoder_do_not_import_torch():
    # Fresh interpreter: importing the pure core, the bounds layer, and the MILP
    # encoder must not pull in torch or the benchmarks package -- that is exactly
    # the dependency direction the Phase 1 split establishes.
    code = (
        "import sys\n"
        "import nnequiv.core\n"
        "import nnequiv.bounds\n"
        "import nn_equivalence.encoder_pyomo\n"
        "assert 'torch' not in sys.modules, 'the pure core pulled in torch'\n"
        "assert 'benchmarks' not in sys.modules, 'the encoder pulled in benchmarks'\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
