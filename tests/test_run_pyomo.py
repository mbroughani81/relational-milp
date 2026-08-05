from typing import Any

import pytest

pyo = pytest.importorskip("pyomo.environ")

from benchmarks.common import (
    HalfSpace,
    Hyperrectangle,
    HPolytope,
    Instance,
)
from benchmarks.run_pyomo import run_instance
from nn_equivalence.encoder_pyomo import (
    add_output_distance_constraint,
    validate_directional_witness,
)


def make_instance(epsilon: float) -> Instance:
    return Instance(
        instance_id="test_instance",
        suite_name="test",
        nn1=[([[1.0]], [0.0])],
        nn2=[([[0.0]], [0.0])],
        input_region=Hyperrectangle(low=[0.0], high=[1.0]),
        epsilon=epsilon,
    )


def make_input_var(value: float) -> list[Any]:
    model = pyo.ConcreteModel()
    model.x = pyo.Var([0], bounds=(0.0, 1.0), initialize=value)
    return [model.x[0]]


def test_validate_directional_witness_accepts_valid_witness(capsys) -> None:
    instance = make_instance(epsilon=0.5)

    validate_directional_witness(
        instance,
        make_input_var(0.6),
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
    )

    assert capsys.readouterr().err == ""


def test_validate_directional_witness_warns_for_invalid_margin(capsys) -> None:
    instance = make_instance(epsilon=0.5)

    validate_directional_witness(
        instance,
        make_input_var(0.2),
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
    )

    err = capsys.readouterr().err
    assert "Solver returned a feasible point" in err
    assert "target_verified=False" in err
    assert "input_verified=True" in err


def test_run_instance_respects_polyhedral_input_constraints() -> None:
    instance = Instance(
        instance_id="polyhedron_instance",
        suite_name="test",
        nn1=[([[1.0, 1.0]], [0.0])],
        nn2=[([[0.0, 0.0]], [0.0])],
        input_region=HPolytope(
            [
                HalfSpace([-1.0, 0.0], 0.0),
                HalfSpace([1.0, 0.0], 1.0),
                HalfSpace([0.0, -1.0], 0.0),
                HalfSpace([0.0, 1.0], 1.0),
                HalfSpace([1.0, 1.0], 1.0),
            ]
        ),
        epsilon=1.5,
    )

    result = run_instance(
        instance,
        solver_name="highs",
        bound_tightening="interval",
        abcrown_bound_cache=None,
    )

    assert result.status == "unsat"


def test_output_distance_constraint_uses_only_selected_output() -> None:
    model = pyo.ConcreteModel()
    model.constraints = pyo.Constraint(pyo.Any)
    model.first = pyo.Var(range(3))
    model.second = pyo.Var(range(3))

    add_output_distance_constraint(
        model.constraints,
        [model.first[index] for index in range(3)],
        [model.second[index] for index in range(3)],
        epsilon=0.5,
        output_index=1,
    )

    assert len(model.constraints) == 1
    expression = str(model.constraints[0].expr)
    assert "first[1]" in expression
    assert "second[1]" in expression
    assert "first[0]" not in expression
    assert "first[2]" not in expression
    assert not any(
        variable.is_binary()
        for variable in model.component_data_objects(pyo.Var)
    )


def test_validate_witness_ignores_non_target_output_violation(capsys) -> None:
    instance = Instance(
        instance_id="target_output_instance",
        suite_name="test",
        nn1=[([[10.0], [1.0]], [0.0, 0.0])],
        nn2=[([[0.0], [0.0]], [0.0, 0.0])],
        input_region=Hyperrectangle(low=[0.0], high=[1.0]),
        epsilon=0.5,
        output_index=1,
    )

    validate_directional_witness(
        instance,
        make_input_var(0.2),
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
    )

    err = capsys.readouterr().err
    assert "target_verified=False" in err
    assert "output_index=1" in err
