import pyomo.environ as pyo

from benchmarks.common import InputRegion, Instance
from nn_equivalence.encoder_pyomo import validate_directional_witness


def make_instance(epsilon: float) -> Instance:
    return Instance(
        instance_id="test_instance",
        suite_name="test",
        nn1=[([[1.0]], [0.0])],
        nn2=[([[0.0]], [0.0])],
        input_region=InputRegion(lower_bounds=[0.0], upper_bounds=[1.0]),
        epsilon=epsilon,
    )


def make_input_var(value: float) -> list[pyo.Var]:
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
