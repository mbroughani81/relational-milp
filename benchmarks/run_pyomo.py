from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC
from pyomo.repn import generate_standard_repn

import nn_equivalence.encoder_pyomo as encoder
from benchmarks.abcrown_bounds import (
    ABCrownBoundCache,
    ABCrownBoundOptions,
    compute_network_bounds,
)
from benchmarks.common import (
    Hyperrectangle,
    Instance,
    InstanceResult,
    InstanceStatus,
    InstanceSuite,
    SolveStats,
    SuiteOptions,
    parse_suite_options,
    validate_instance,
)
from nn_equivalence.nn_types import Bounds, NeuralNetwork

BoundTighteningMode = Literal["interval", "abcrown"]
SolverName = Literal["highs", "gurobi", "cplex"]


@dataclass(frozen=True)
class BoundResult:
    bounds: encoder.NetworkBounds
    nn1_runtime_sec: float
    nn2_runtime_sec: float


@dataclass(frozen=True)
class DirectionResult:
    status: InstanceStatus
    stats: SolveStats


@dataclass(frozen=True)
class ModelBinaryStats:
    rows: int | str
    cols: int | str
    nonzeros: int | str
    all_binary_variables: int | str
    unfixed_binary_variables: int | str


@dataclass(frozen=True)
class CplexDebugStats:
    loaded_model_stats: ModelBinaryStats | None
    after_presolve_stats: ModelBinaryStats | None
    progress_details: list[tuple[str, str | int | float]]
    presolve_runtime_sec: float


def load_suite(name: str, suite_options: SuiteOptions) -> InstanceSuite:
    module = importlib.import_module(f"benchmarks.{name}")
    return module.load_suite(suite_options)


def format_expected(result: InstanceResult) -> str:
    if result.expected_status is None:
        return ""
    matched = "yes" if result.matched_expected else "no"
    return f"{result.expected_status}:{matched}"


def results_csv(results: list[InstanceResult]) -> str:
    lines = ["instance_id,status,expected,runtime_sec,epsilon"]
    for result in results:
        lines.append(
            f"{result.instance_id},"
            f"{result.status},"
            f"{format_expected(result)},"
            f"{result.runtime_sec:.6f},"
            f"{result.epsilon:.17g}"
        )
    return "\n".join(lines) + "\n"


def print_results(results: list[InstanceResult]) -> None:
    print(results_csv(results), end="")


def write_results_csv(path: Path, results: list[InstanceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_csv(results), encoding="utf-8")


def write_debug_json(path: Path, debug_payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(debug_payloads, indent=2) + "\n", encoding="utf-8")


def filter_instances(
    instances: list[Instance],
    limit: int | None,
    ids: list[str],
) -> list[Instance]:
    if limit is not None:
        if limit < 1:
            raise ValueError("suite option 'limit' must be at least 1")
        return instances[:limit]
    if not ids:
        raise ValueError("either suite option 'limit' or 'ids' must be provided")

    selected_ids = set(ids)
    selected_instances = [
        instance for instance in instances if instance.instance_id in selected_ids
    ]
    missing_ids = selected_ids - {
        instance.instance_id for instance in selected_instances
    }
    if missing_ids:
        raise ValueError(f"unknown instance ids: {sorted(missing_ids)}")
    return selected_instances


def parse_instance_ids(raw_ids: list[str]) -> list[str]:
    return [
        instance_id.strip()
        for raw_id in raw_ids
        for instance_id in raw_id.split(",")
        if instance_id.strip()
    ]


def extract_selection_from_suite_options(
    suite_options: SuiteOptions,
) -> tuple[SuiteOptions, int | None, list[str]]:
    options = dict(suite_options)
    suite_limit = options.pop("limit", None)
    suite_ids = options.pop("ids", None)
    if suite_limit is not None and suite_ids is not None:
        raise ValueError("suite options 'limit' and 'ids' cannot be combined")
    if suite_limit is not None:
        return options, int(suite_limit), []
    if suite_ids is not None:
        return options, None, parse_instance_ids([suite_ids])
    return options, None, []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an NN equivalence instance suite with Pyomo."
    )
    parser.add_argument("--suite", default="sample")
    parser.add_argument(
        "--solver",
        default="highs",
        choices=("highs", "gurobi", "cplex"),
    )
    parser.add_argument(
        "--suite-options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Suite-specific option. Repeat for multiple options; values may "
            "contain commas, e.g. --suite-options modes=global,three_pixel."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print backend solver logs to stdout.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print CPLEX-only structured model-size, ReLU-binary, presolve, "
            "and solver-progress details."
        ),
    )
    parser.add_argument(
        "--debug-out",
        type=Path,
        default=None,
        help="Also write CPLEX debug JSON to this file.",
    )
    parser.add_argument(
        "--bound-tightening",
        default="interval",
        choices=("interval", "abcrown"),
        help=(
            "Bound source for Pyomo ReLU encodings. 'interval' uses interval "
            "arithmetic; 'abcrown' tightens those bounds with certified "
            "alpha-beta-CROWN compute_bounds results."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write the final CSV results to this file.",
    )
    return parser.parse_args()


def pyomo_solver_name(solver_name: SolverName) -> str:
    if solver_name == "cplex":
        return "cplex_direct"
    return solver_name


def set_solver_timeout(
    solver: Any, solver_name: SolverName, timeout_sec: float
) -> None:
    if hasattr(solver, "options"):
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = timeout_sec
        elif solver_name == "cplex":
            solver.options["timelimit"] = timeout_sec
        else:
            solver.options["time_limit"] = timeout_sec
    elif hasattr(solver, "config") and hasattr(solver.config, "time_limit"):
        solver.config.time_limit = timeout_sec


def create_solver(
    solver_name: SolverName,
    timeout_sec: float,
) -> Any:
    backend_name = pyomo_solver_name(solver_name)
    solver = pyo.SolverFactory(backend_name)
    if not solver.available(False):
        raise RuntimeError(
            f"Pyomo solver '{solver_name}' is not available through backend "
            f"'{backend_name}'. Install the solver backend and make it "
            "available to Pyomo."
        )

    set_solver_timeout(solver, solver_name, timeout_sec)
    return solver


def count_pyomo_nonzeros(model: pyo.ConcreteModel) -> int | str:
    nonzeros = 0
    for constraint in model.component_data_objects(
        pyo.Constraint,
        active=True,
        descend_into=True,
    ):
        representation = generate_standard_repn(constraint.body)
        if not representation.is_linear():
            return "nonlinear"
        nonzeros += len(representation.linear_vars)
    return nonzeros


def pyomo_model_stats(model: pyo.ConcreteModel) -> ModelBinaryStats:
    variables = list(
        model.component_data_objects(pyo.Var, active=True, descend_into=True)
    )
    binary_variables = [variable for variable in variables if variable.is_binary()]
    rows = sum(
        1
        for _ in model.component_data_objects(
            pyo.Constraint,
            active=True,
            descend_into=True,
        )
    )
    return ModelBinaryStats(
        rows=rows,
        cols=len(variables),
        nonzeros=count_pyomo_nonzeros(model),
        all_binary_variables=len(binary_variables),
        unfixed_binary_variables=sum(
            1 for variable in binary_variables if not variable.fixed
        ),
    )


def cplex_loaded_model_stats(solver: Any) -> ModelBinaryStats | None:
    solver_model = getattr(solver, "_solver_model", None)
    if solver_model is None or not hasattr(solver_model, "variables"):
        return None

    variable_types = list(solver_model.variables.get_types())
    lower_bounds = list(solver_model.variables.get_lower_bounds())
    upper_bounds = list(solver_model.variables.get_upper_bounds())
    binary_count = 0
    unfixed_binary_count = 0
    for variable_type, lower_bound, upper_bound in zip(
        variable_types,
        lower_bounds,
        upper_bounds,
    ):
        if variable_type == solver_model.variables.type.binary:
            binary_count += 1
            if lower_bound != upper_bound:
                unfixed_binary_count += 1

    return ModelBinaryStats(
        rows=solver_model.linear_constraints.get_num(),
        cols=solver_model.variables.get_num(),
        nonzeros=solver_model.linear_constraints.get_num_nonzeros(),
        all_binary_variables=binary_count,
        unfixed_binary_variables=unfixed_binary_count,
    )


def cplex_after_presolve_stats(solver: Any) -> ModelBinaryStats | None:
    solver_model = getattr(solver, "_solver_model", None)
    if solver_model is None or not hasattr(solver_model, "presolve"):
        return None

    solver_model.presolve.presolve(solver_model.presolve.method.primal)
    presolved_col_status = solver_model.presolve.get_presolved_col_status()
    presolved_row_status = solver_model.presolve.get_presolved_row_status()
    variable_types = list(solver_model.variables.get_types())
    lower_bounds = list(solver_model.variables.get_lower_bounds())
    upper_bounds = list(solver_model.variables.get_upper_bounds())

    binary_count = 0
    unfixed_binary_count = 0
    for original_index in presolved_col_status:
        if original_index < 0:
            continue

        is_binary = variable_types[original_index] == solver_model.variables.type.binary
        unfixed = lower_bounds[original_index] != upper_bounds[original_index]
        if is_binary:
            binary_count += 1
            if unfixed:
                unfixed_binary_count += 1

    stats = ModelBinaryStats(
        rows=len(presolved_row_status),
        cols=len(presolved_col_status),
        nonzeros="unavailable",
        all_binary_variables=binary_count,
        unfixed_binary_variables=unfixed_binary_count,
    )
    return stats


def cplex_progress_details(solver: Any) -> list[tuple[str, str | int | float]]:
    solver_model = getattr(solver, "_solver_model", None)
    if solver_model is None or not hasattr(solver_model, "solution"):
        return []

    details: list[tuple[str, str | int | float]] = []
    progress = solver_model.solution.progress
    for output_name, method_name in (
        ("nodes_processed", "get_num_nodes_processed"),
        ("nodes_remaining", "get_num_nodes_remaining"),
        ("iterations", "get_num_iterations"),
    ):
        if hasattr(progress, method_name):
            details.append((output_name, getattr(progress, method_name)()))
    mip_solution = solver_model.solution.MIP
    if (
        hasattr(mip_solution, "get_mip_relative_gap")
        and solver_model.solution.is_primal_feasible()
    ):
        try:
            details.append(("mip_relative_gap", mip_solution.get_mip_relative_gap()))
        except Exception:
            details.append(("mip_relative_gap", "unavailable"))
    return details


def cplex_debug_stats(solver: Any) -> CplexDebugStats:
    loaded_model_stats = cplex_loaded_model_stats(solver)

    presolve_start = time.perf_counter()
    after_presolve_stats = cplex_after_presolve_stats(solver)
    presolve_runtime_sec = time.perf_counter() - presolve_start
    progress_details = cplex_progress_details(solver)

    return CplexDebugStats(
        loaded_model_stats=loaded_model_stats,
        after_presolve_stats=after_presolve_stats,
        progress_details=progress_details,
        presolve_runtime_sec=presolve_runtime_sec,
    )


def status_from_pyomo(termination_condition: TC) -> InstanceStatus:
    if termination_condition in {TC.optimal, TC.feasible, TC.globallyOptimal}:
        return "sat"
    if termination_condition == TC.infeasible:
        return "unsat"
    if termination_condition == TC.maxTimeLimit:
        return "timeout"
    return "unknown"


def model_stats_details(
    phase: str,
    stats: ModelBinaryStats,
) -> list[tuple[str, str | int | float]]:
    return [
        (f"{phase}_rows", stats.rows),
        (f"{phase}_cols", stats.cols),
        (f"{phase}_nonzeros", stats.nonzeros),
        (f"{phase}_all_binary_variables", stats.all_binary_variables),
        (f"{phase}_unfixed_binary_variables", stats.unfixed_binary_variables),
    ]


def format_value(value: str | int | float) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_detail_pairs(details: list[tuple[str, str | int | float]]) -> str:
    return " ".join(f"{name}={format_value(value)}" for name, value in details)


def direction_debug_details(
    encoding_stats: encoder.EncodingDebugStats,
    before_presolve_stats: ModelBinaryStats,
    cplex_stats: CplexDebugStats,
) -> list[tuple[str, str | int | float]]:
    details: list[tuple[str, str | int | float]] = [
        ("direction", encoding_stats.direction_name),
        (
            "output_selector_binary_variables",
            encoding_stats.output_selector_binary_variables,
        ),
        ("all_binary_variables", encoding_stats.all_binary_variables),
        ("unfixed_binary_variables", encoding_stats.unfixed_binary_variables),
    ]

    details.extend(model_stats_details("before_presolve", before_presolve_stats))
    if cplex_stats.loaded_model_stats is None:
        details.append(("cplex_loaded_model_stats", "unavailable"))
    else:
        details.extend(
            model_stats_details(
                "cplex_loaded_model",
                cplex_stats.loaded_model_stats,
            )
        )
    if cplex_stats.after_presolve_stats is None:
        details.append(("after_presolve_stats", "unavailable"))
    else:
        details.extend(
            model_stats_details(
                "after_presolve",
                cplex_stats.after_presolve_stats,
            )
        )

    details.extend(cplex_stats.progress_details)
    return details


def solve_instance_direction(
    instance: Instance,
    solver_name: SolverName,
    first_network_name: str,
    second_network_name: str,
    first_network: NeuralNetwork,
    second_network: NeuralNetwork,
    verbose: bool,
    debug: bool,
    bounds: encoder.NetworkBounds,
) -> DirectionResult:
    encode_start = time.perf_counter()
    encoded = encoder.encode_instance_direction(
        instance,
        first_network_name,
        second_network_name,
        first_network,
        second_network,
        bounds,
    )
    encode_runtime_sec = time.perf_counter() - encode_start
    model = encoded.model
    input_vars = encoded.input_vars
    before_presolve_stats = None
    if debug:
        before_presolve_stats = pyomo_model_stats(model)

    solver_setup_start = time.perf_counter()
    solver = create_solver(
        solver_name,
        instance.timeout_sec,
    )
    solver_setup_runtime_sec = time.perf_counter() - solver_setup_start

    start_time = time.perf_counter()
    result = solver.solve(
        model,
        tee=verbose,
        load_solutions=False,
    )
    runtime_sec = time.perf_counter() - start_time
    status = status_from_pyomo(result.solver.termination_condition)

    if status == "sat":
        model.solutions.load_from(result)
        encoder.validate_directional_witness(
            instance,
            input_vars,
            first_network_name,
            second_network_name,
            first_network,
            second_network,
        )

    direction_name = f"{first_network_name}_minus_{second_network_name}"
    details: list[tuple[str, str | int | float]] = []
    debug_timings: list[tuple[str, float]] = []
    if debug:
        if solver_name != "cplex":
            raise RuntimeError(
                "--debug is currently supported only with --solver cplex"
            )
        if before_presolve_stats is None:
            raise ValueError("debug enabled without before-presolve stats")
        cplex_stats = cplex_debug_stats(solver)
        details = direction_debug_details(
            encoded.debug_stats,
            before_presolve_stats,
            cplex_stats,
        )
        debug_timings = [
            ("presolve", cplex_stats.presolve_runtime_sec),
        ]
    timings = [
        ("encode", encode_runtime_sec),
        ("solver_setup", solver_setup_runtime_sec),
        ("solve", runtime_sec),
    ]
    timings.extend(debug_timings)
    return DirectionResult(
        status=status,
        stats=SolveStats(
            name=direction_name,
            timings=timings,
            details=details,
        ),
    )


def combine_directional_statuses(statuses: list[InstanceStatus]) -> InstanceStatus:
    if "sat" in statuses:
        return "sat"
    if all(status == "unsat" for status in statuses):
        return "unsat"
    if "timeout" in statuses:
        return "timeout"
    return "unknown"


def run_instance(
    instance: Instance,
    solver_name: SolverName,
    bound_tightening: BoundTighteningMode,
    abcrown_bound_cache: ABCrownBoundCache | None,
    verbose: bool = False,
    debug: bool = False,
) -> InstanceResult:
    if debug and solver_name != "cplex":
        raise RuntimeError("--debug is currently supported only with --solver cplex")
    validate_instance(instance)
    bound_result = compute_bounds(
        instance,
        bound_tightening,
        abcrown_bound_cache,
    )
    bounds = bound_result.bounds

    first_result = solve_instance_direction(
        instance,
        solver_name,
        "nn1",
        "nn2",
        instance.nn1,
        instance.nn2,
        verbose,
        debug,
        bounds,
    )
    second_result = solve_instance_direction(
        instance,
        solver_name,
        "nn2",
        "nn1",
        instance.nn2,
        instance.nn1,
        verbose,
        debug,
        bounds,
    )
    status = combine_directional_statuses([first_result.status, second_result.status])
    stats = [
        SolveStats(
            name="bound_tightening",
            timings=[
                ("nn1", bound_result.nn1_runtime_sec),
                ("nn2", bound_result.nn2_runtime_sec),
            ],
        ),
        first_result.stats,
        second_result.stats,
    ]

    return InstanceResult(
        instance_id=instance.instance_id,
        suite_name=instance.suite_name,
        status=status,
        runtime_sec=(
            bound_result.nn1_runtime_sec
            + bound_result.nn2_runtime_sec
            + timing_total(stats, "solve")
        ),
        epsilon=instance.epsilon,
        expected_status=instance.expected_status,
        stats=stats,
    )


def affine_bounds(
    weights: list[list[float]],
    bias: list[float],
    input_bounds: Bounds,
) -> Bounds:
    output_bounds: Bounds = []

    for row, bias_value in zip(weights, bias):
        lower = bias_value
        upper = bias_value
        for weight, (input_lower, input_upper) in zip(row, input_bounds):
            if weight >= 0:
                lower += weight * input_lower
                upper += weight * input_upper
            else:
                lower += weight * input_upper
                upper += weight * input_lower
        output_bounds.append((lower, upper))

    return output_bounds


def relu_bounds(z_bounds: Bounds) -> Bounds:
    return [(max(0.0, lower), max(0.0, upper)) for lower, upper in z_bounds]


def tighten_bounds(interval_bounds: Bounds, bound: Bounds | None) -> Bounds:
    if bound is None:
        return interval_bounds
    if len(interval_bounds) != len(bound):
        raise ValueError("bound length does not match interval bounds")

    tightened: Bounds = []
    for (interval_lower, interval_upper), (bound_lower, bound_upper) in zip(
        interval_bounds,
        bound,
    ):
        lower = max(interval_lower, bound_lower)
        upper = min(interval_upper, bound_upper)
        if lower > upper:
            if lower - upper <= 1e-8:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            else:
                raise ValueError(
                    "bound is inconsistent with interval bounds: "
                    f"interval=({interval_lower}, {interval_upper}), "
                    f"bound=({bound_lower}, {bound_upper})"
                )
        tightened.append((lower, upper))
    return tightened


def compute_interval_bounds(
    network: NeuralNetwork,
    input_bounds: Bounds,
    bounds: list[Bounds] | None = None,
) -> list[Bounds]:
    if not network:
        raise ValueError("neural network must have at least one layer")
    if bounds is not None and len(bounds) != len(network):
        raise ValueError("bound layer count does not match network")

    network_bounds: list[Bounds] = []
    current_bounds = input_bounds
    for layer_index, (weights, bias) in enumerate(network):
        interval_z_bounds = affine_bounds(weights, bias, current_bounds)
        bound = None if bounds is None else bounds[layer_index]
        z_bounds = tighten_bounds(interval_z_bounds, bound)
        network_bounds.append(z_bounds)
        if layer_index != len(network) - 1:
            current_bounds = relu_bounds(z_bounds)

    return network_bounds


def compute_bounds(
    instance: Instance,
    bound_tightening: BoundTighteningMode,
    abcrown_bound_cache: ABCrownBoundCache | None,
) -> BoundResult:
    input_box = Hyperrectangle.overapproximate(instance.input_region)
    input_bounds: Bounds = input_box.bounds()

    if bound_tightening == "interval":
        nn1_start = time.perf_counter()
        nn1_bounds = compute_interval_bounds(instance.nn1, input_bounds)
        nn1_runtime_sec = time.perf_counter() - nn1_start

        nn2_start = time.perf_counter()
        nn2_bounds = compute_interval_bounds(instance.nn2, input_bounds)
        nn2_runtime_sec = time.perf_counter() - nn2_start

        return BoundResult(
            bounds={
                "nn1": nn1_bounds,
                "nn2": nn2_bounds,
            },
            nn1_runtime_sec=nn1_runtime_sec,
            nn2_runtime_sec=nn2_runtime_sec,
        )
    if bound_tightening != "abcrown":
        raise ValueError(f"unsupported bound tightening mode: {bound_tightening}")

    options = ABCrownBoundOptions(timeout_sec=instance.timeout_sec)

    nn1_start = time.perf_counter()
    nn1_abcrown_bounds = compute_network_bounds(
        instance.nn1,
        input_bounds,
        options,
        abcrown_bound_cache,
    )
    nn1_bounds = compute_interval_bounds(
        instance.nn1,
        input_bounds,
        nn1_abcrown_bounds,
    )
    nn1_runtime_sec = time.perf_counter() - nn1_start

    nn2_start = time.perf_counter()
    nn2_abcrown_bounds = compute_network_bounds(
        instance.nn2,
        input_bounds,
        options,
        abcrown_bound_cache,
    )
    nn2_bounds = compute_interval_bounds(
        instance.nn2,
        input_bounds,
        nn2_abcrown_bounds,
    )
    nn2_runtime_sec = time.perf_counter() - nn2_start

    return BoundResult(
        bounds={
            "nn1": nn1_bounds,
            "nn2": nn2_bounds,
        },
        nn1_runtime_sec=nn1_runtime_sec,
        nn2_runtime_sec=nn2_runtime_sec,
    )


def timing_total(stats: list[SolveStats], phase_name: str) -> float:
    return sum(
        runtime_sec
        for solve_stats in stats
        for phase, runtime_sec in solve_stats.timings
        if phase == phase_name
    )


def format_solve_stats(stats: list[SolveStats]) -> str:
    parts = []
    for solve_stats in stats:
        timing_text = ",".join(
            f"{phase}={runtime_sec:.3f}" for phase, runtime_sec in solve_stats.timings
        )
        parts.append(f"{solve_stats.name}[{timing_text}]")
    measured_total_sec = sum(solve_stats.measured_total_sec for solve_stats in stats)
    parts.append(f"total={measured_total_sec:.3f}")
    return " ".join(parts)


def detail_value(
    details: dict[str, str | int | float],
    key: str,
) -> str | int | float | None:
    return details.get(key)


def model_stats_json(
    details: dict[str, str | int | float],
    prefix: str,
) -> dict[str, str | int | float | None]:
    return {
        "rows": detail_value(details, f"{prefix}_rows"),
        "cols": detail_value(details, f"{prefix}_cols"),
        "nonzeros": detail_value(details, f"{prefix}_nonzeros"),
        "all_binary_variables": detail_value(
            details,
            f"{prefix}_all_binary_variables",
        ),
        "unfixed_binary_variables": detail_value(
            details,
            f"{prefix}_unfixed_binary_variables",
        ),
    }


def solve_stats_debug_json(solve_stats: SolveStats) -> dict[str, Any]:
    details = dict(solve_stats.details)
    return {
        "direction": details.get("direction", solve_stats.name),
        "timings_sec": {
            phase: runtime_sec for phase, runtime_sec in solve_stats.timings
        },
        "total": {
            "all_binary_variables": detail_value(
                details,
                "all_binary_variables",
            ),
            "unfixed_binary_variables": detail_value(
                details,
                "unfixed_binary_variables",
            ),
            "output_selector_binary_variables": detail_value(
                details,
                "output_selector_binary_variables",
            ),
        },
        "before_presolve": model_stats_json(details, "before_presolve"),
        "cplex_loaded_model": model_stats_json(
            details,
            "cplex_loaded_model",
        ),
        "after_presolve": model_stats_json(details, "after_presolve"),
        "cplex_progress": {
            name: value
            for name, value in details.items()
            if name
            in {
                "nodes_processed",
                "nodes_remaining",
                "iterations",
                "mip_relative_gap",
            }
        },
    }


def phase_timings_json(stats: list[SolveStats]) -> dict[str, Any]:
    return {
        "phases": {
            solve_stats.name: {
                phase: runtime_sec for phase, runtime_sec in solve_stats.timings
            }
            for solve_stats in stats
        },
        "measured_total_sec": sum(
            solve_stats.measured_total_sec for solve_stats in stats
        ),
    }


def instance_debug_json(result: InstanceResult) -> dict[str, Any]:
    directions: dict[str, dict[str, Any]] = {}
    for solve_stats in result.stats:
        if not solve_stats.details:
            continue
        direction_payload = solve_stats_debug_json(solve_stats)
        direction = str(direction_payload["direction"])
        directions[direction] = direction_payload

    return {
        "type": "pyomo_cplex_instance_debug",
        "instance_id": result.instance_id,
        "suite_name": result.suite_name,
        "status": result.status,
        "expected": format_expected(result),
        "runtime_sec": result.runtime_sec,
        "epsilon": result.epsilon,
        "phase_timings_sec": phase_timings_json(result.stats),
        "directions": directions,
    }


def print_progress(
    index: int,
    total: int,
    result: InstanceResult,
    debug_payload: dict[str, Any] | None,
) -> None:
    phase_text = ""
    if result.stats:
        phase_text = f" phases: {format_solve_stats(result.stats)}"
    print(
        f"[{index}/{total}] {result.instance_id}: "
        f"status={result.status} expected={format_expected(result) or '-'} "
        f"runtime_sec={result.runtime_sec:.3f} epsilon={result.epsilon:.17g}"
        f"{phase_text}",
        flush=True,
    )
    if debug_payload is not None:
        print(json.dumps(debug_payload, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    try:
        debug_enabled = args.debug or args.debug_out is not None
        suite_options, limit, ids = extract_selection_from_suite_options(
            parse_suite_options(args.suite_options),
        )
        suite = load_suite(args.suite, suite_options)
        instances = filter_instances(
            suite.instances,
            limit,
            ids,
        )
        abcrown_bound_cache = (
            ABCrownBoundCache() if args.bound_tightening == "abcrown" else None
        )
        results: list[InstanceResult] = []
        debug_payloads: list[dict[str, Any]] = []
        total_instances = len(instances)
        for index, instance in enumerate(instances, start=1):
            result = run_instance(
                instance,
                args.solver,
                args.bound_tightening,
                abcrown_bound_cache,
                args.verbose,
                debug_enabled,
            )
            results.append(result)
            debug_payload = instance_debug_json(result) if debug_enabled else None
            if args.debug_out is not None and debug_payload is not None:
                debug_payloads.append(debug_payload)
            print_progress(
                index,
                total_instances,
                result,
                debug_payload if args.debug else None,
            )
    except (RuntimeError, ValueError) as error:
        print(error)
        raise SystemExit(2) from error

    print_results(results)
    if args.csv is not None:
        write_results_csv(args.csv, results)
    if args.debug_out is not None:
        write_debug_json(args.debug_out, debug_payloads)


if __name__ == "__main__":
    main()
