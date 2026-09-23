"""Tests for the linf and top1 MILP encodings (nn_equivalence/encoder_pyomo.py).

Every network here is hand-built and tiny, so the expected answer is checked by
hand rather than against another verifier. Models are solved with HiGHS, which
ships with the Python requirements, instead of going through
``run_pyomo.create_solver`` (that always reaches for CPLEX).
"""

import pyomo.environ as pyo
import pytest

import nn_equivalence.encoder_pyomo as encoder
from benchmarks.common import Hyperrectangle, Instance
from benchmarks.run_pyomo import compute_bounds

# 2 inputs -> 2 outputs, one affine layer, so outputs are exactly the inputs.
IDENTITY = [([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])]
# Same argmax as IDENTITY everywhere (positive scaling), different values.
SCALED = [([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0])]
# Argmax flipped relative to IDENTITY.
SWAPPED = [([[0.0, 1.0], [1.0, 0.0]], [0.0, 0.0])]
# Identical to IDENTITY on output 0, +5 on output 1.
SHIFTED_OUTPUT_1 = [([[1.0, 0.0], [0.0, 1.0]], [0.0, 5.0])]

UNIT_BOX = Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0])


def make_instance(nn1, nn2, property_kind, epsilon, *, output_index=0, region=UNIT_BOX):
    return Instance(
        instance_id=f"test_{property_kind}",
        suite_name="test",
        nn1=nn1,
        nn2=nn2,
        input_region=region,
        epsilon=epsilon,
        output_index=output_index,
        property_kind=property_kind,
        timeout_sec=30.0,
    )


def _solve_model(encoded):
    result = pyo.SolverFactory("appsi_highs").solve(
        encoded.model, load_solutions=False
    )
    condition = str(result.solver.termination_condition)
    if condition in ("optimal", "feasible"):
        encoded.model.solutions.load_from(result)
        return "sat"
    assert condition == "infeasible", condition
    return "unsat"


def solve(instance):
    """Return ("sat"/"unsat", encoded) for the instance, as run_pyomo would.

    The epsilon properties are asymmetric, so -- like
    ``run_pyomo.run_instance`` -- both directions are solved and a violation in
    either one counts. top1 is symmetric and needs a single model.
    """
    bounds = compute_bounds(instance, "interval").bounds
    if instance.property_kind == "top1":
        encoded = encoder.encode_instance_top1(instance, "nn1", "nn2", bounds)
        return _solve_model(encoded), encoded

    first_encoded = None
    for first, second, first_net, second_net in (
        ("nn1", "nn2", instance.nn1, instance.nn2),
        ("nn2", "nn1", instance.nn2, instance.nn1),
    ):
        encoded = encoder.encode_instance_direction(
            instance, first, second, first_net, second_net, bounds
        )
        first_encoded = first_encoded or encoded
        if _solve_model(encoded) == "sat":
            return "sat", encoded
    return "unsat", first_encoded


def status(instance):
    return solve(instance)[0]


# --------------------------------------------------------------------------- #
# linf
# --------------------------------------------------------------------------- #


def test_linf_finds_violation_on_a_non_labeled_output():
    """The whole point of linf: logit_class on output 0 misses this."""
    assert status(make_instance(IDENTITY, SHIFTED_OUTPUT_1, "logit_class", 1.0)) == "unsat"
    assert status(make_instance(IDENTITY, SHIFTED_OUTPUT_1, "linf", 1.0)) == "sat"


def test_linf_respects_the_epsilon_threshold():
    # The only gap is exactly 5, on output 1.
    assert status(make_instance(IDENTITY, SHIFTED_OUTPUT_1, "linf", 4.9)) == "sat"
    assert status(make_instance(IDENTITY, SHIFTED_OUTPUT_1, "linf", 6.0)) == "unsat"


def test_linf_on_identical_networks_is_unsat():
    assert status(make_instance(IDENTITY, IDENTITY, "linf", 0.5)) == "unsat"


def test_linf_adds_one_selector_binary_per_output_and_logit_class_adds_none():
    _, linf = solve(make_instance(IDENTITY, SHIFTED_OUTPUT_1, "linf", 1.0))
    assert linf.debug_stats.output_selector_binary_variables == 2

    _, single = solve(make_instance(IDENTITY, SHIFTED_OUTPUT_1, "logit_class", 1.0))
    assert single.debug_stats.output_selector_binary_variables == 0


def test_multi_output_distance_constraint_requires_bounds_for_its_big_m():
    model = pyo.ConcreteModel()
    with pytest.raises(ValueError, match="output bounds are required"):
        encoder.add_output_distance_constraint(
            model,
            pyo.Constraint(pyo.Any),
            [pyo.Var(), pyo.Var()],
            [pyo.Var(), pyo.Var()],
            1.0,
            [0, 1],
        )


# --------------------------------------------------------------------------- #
# top1
# --------------------------------------------------------------------------- #


def test_top1_is_unsat_for_networks_that_share_an_argmax():
    assert status(make_instance(IDENTITY, IDENTITY, "top1", 1e-4)) == "unsat"
    # A positive rescaling preserves argmax everywhere, including at ties.
    assert status(make_instance(IDENTITY, SCALED, "top1", 1e-4)) == "unsat"


def test_top1_finds_a_genuine_argmax_disagreement():
    assert status(make_instance(IDENTITY, SWAPPED, "top1", 1e-4)) == "sat"
    assert status(make_instance(IDENTITY, SHIFTED_OUTPUT_1, "top1", 1e-4)) == "sat"


def test_top1_witness_reproduces_numerically():
    instance = make_instance(IDENTITY, SWAPPED, "top1", 1e-4)
    outcome, encoded = solve(instance)
    assert outcome == "sat"
    # Raises only on an uninitialized witness; a mismatch would print a warning,
    # so assert the counterexample directly instead.
    encoder.validate_top1_witness(instance, encoded.input_vars)
    point = [float(pyo.value(var)) for var in encoded.input_vars]
    first = encoder.forward_values(instance.nn1, point)
    second = encoder.forward_values(instance.nn2, point)
    assert first.index(max(first)) != second.index(max(second))


def test_top1_rejects_a_margin_at_or_below_the_solver_tolerance():
    """A margin of 0 would make every tie point a counterexample."""
    for margin in (0.0, 1e-9):
        with pytest.raises(ValueError, match="strictness margin"):
            solve(make_instance(IDENTITY, SCALED, "top1", margin))


def test_top1_encoding_is_one_model_with_two_selector_sets():
    _, encoded = solve(make_instance(IDENTITY, SWAPPED, "top1", 1e-4))
    assert encoded.debug_stats.direction_name == "top1_disagreement"
    # One binary per output, per network.
    assert encoded.debug_stats.output_selector_binary_variables == 4


def test_encode_instance_top1_rejects_other_properties():
    with pytest.raises(ValueError, match="requires property_kind 'top1'"):
        instance = make_instance(IDENTITY, SWAPPED, "linf", 1.0)
        encoder.encode_instance_top1(
            instance, "nn1", "nn2", compute_bounds(instance, "interval").bounds
        )


def test_encode_instance_direction_rejects_top1():
    instance = make_instance(IDENTITY, SWAPPED, "top1", 1e-4)
    with pytest.raises(ValueError, match="not a distance property"):
        encoder.encode_instance_direction(
            instance,
            "nn1",
            "nn2",
            instance.nn1,
            instance.nn2,
            compute_bounds(instance, "interval").bounds,
        )


# --------------------------------------------------------------------------- #
# Instance plumbing
# --------------------------------------------------------------------------- #


def test_output_indices_per_property():
    assert make_instance(IDENTITY, SWAPPED, "linf", 1.0).output_indices == (0, 1)
    assert make_instance(
        IDENTITY, SWAPPED, "logit_class", 1.0, output_index=1
    ).output_indices == (1,)
    assert make_instance(IDENTITY, SWAPPED, "top1", 1e-4).output_indices is None


def test_property_kind_defaults_to_the_legacy_single_output_property():
    instance = Instance(
        instance_id="default",
        suite_name="test",
        nn1=IDENTITY,
        nn2=SWAPPED,
        input_region=UNIT_BOX,
        epsilon=1.0,
    )
    assert instance.property_kind == "logit_class"
    assert instance.output_indices == (0,)
