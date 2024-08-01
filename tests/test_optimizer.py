from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.enums import ConstraintType, EventType, OptimizerExitCode
from ropt.plan import OptimizationPlanRunner
from ropt.results import FunctionResults, GradientResults

if TYPE_CHECKING:
    from ropt.optimization import Event


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_max_functions_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def track_results(event: Event) -> None:
        nonlocal last_evaluation
        assert event.results
        last_evaluation = event.results[0].result_id

    max_functions = 2
    enopt_config["optimizer"]["max_functions"] = max_functions
    optimizer = OptimizationPlanRunner(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, track_results
    )
    optimizer.run()
    assert last_evaluation == max_functions
    assert optimizer.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED


def test_max_functions_not_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def track_results(event: Event) -> None:
        nonlocal last_evaluation
        assert event.results
        last_evaluation = event.results[0].result_id

    max_functions = 100
    enopt_config["optimizer"]["max_functions"] = max_functions
    enopt_config["optimizer"]["split_evaluations"] = True
    optimizer = OptimizationPlanRunner(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, track_results
    )
    optimizer.run()
    assert last_evaluation + 1 < 2 * max_functions
    assert optimizer.exit_code == OptimizerExitCode.OPTIMIZER_STEP_FINISHED


def test_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    def _observer(event: Event) -> None:
        assert event.results
        assert isinstance(event.results[0], FunctionResults)
        assert event.results[0].functions is None

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = OptimizationPlanRunner(enopt_config, evaluator(functions)).add_observer(
        EventType.FINISHED_EVALUATION, _observer
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_all_failed_realizations_not_supported(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["realizations"] = {"realization_min_success": 0}

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = OptimizationPlanRunner(
        enopt_config,
        evaluator(functions),
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_user_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def _observer(event: Event) -> None:
        nonlocal last_evaluation
        assert event.results
        last_evaluation = event.results[0].result_id
        if event.results[0].result_id == 1:
            optimizer.abort_optimization()

    optimizer = OptimizationPlanRunner(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _observer
    )
    optimizer.run()
    assert optimizer.results is not None
    assert last_evaluation == 1
    assert optimizer.exit_code == OptimizerExitCode.USER_ABORT


def test_single_perturbation(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["gradient"] = {
        "number_of_perturbations": 1,
        "merge_realizations": True,
    }
    enopt_config["realizations"] = {"weights": 5 * [1]}
    variables = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_objective_auto_scale(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    init1 = test_functions[1](config.variables.initial_values, None)

    enopt_config["objective_functions"]["scales"] = [1.0, init1]
    enopt_config["objective_functions"]["auto_scale"] = False
    manual_result = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert manual_result is not None

    enopt_config["objective_functions"]["scales"] = [1.0, 1.0]
    enopt_config["objective_functions"]["auto_scale"] = [False, True]
    variables = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, manual_result)

    enopt_config["objective_functions"]["scales"] = [1.0, 2.0 * init1]
    enopt_config["objective_functions"]["auto_scale"] = False
    manual_result = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert manual_result is not None

    enopt_config["objective_functions"]["scales"] = [1.0, 2.0]
    enopt_config["objective_functions"]["auto_scale"] = [False, True]
    variables = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, manual_result)


def test_constraint_auto_scale(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4,
        "types": ConstraintType.GE,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0] + variables[2]),
    )

    config = EnOptConfig.model_validate(enopt_config)
    scales = np.fabs(test_functions[-1](config.variables.initial_values, None))

    def check_constraints(event: Event) -> None:
        assert event.results
        for item in event.results:
            if isinstance(item, FunctionResults) and item.result_id == 0:
                assert item.functions is not None
                assert item.functions.scaled_constraints is not None
                assert np.allclose(item.functions.scaled_constraints, 1.0)

    enopt_config["nonlinear_constraints"]["scales"] = scales
    enopt_config["nonlinear_constraints"]["auto_scale"] = False
    OptimizationPlanRunner(enopt_config, evaluator(test_functions)).add_observer(
        EventType.FINISHED_EVALUATION, check_constraints
    ).run()

    enopt_config["nonlinear_constraints"]["scales"] = 1.0
    enopt_config["nonlinear_constraints"]["auto_scale"] = True
    OptimizationPlanRunner(enopt_config, evaluator(test_functions)).add_observer(
        EventType.FINISHED_EVALUATION, check_constraints
    ).run()


@pytest.mark.parametrize("offsets", [None, np.array([1.0, 1.1, 1.2])])
@pytest.mark.parametrize("scales", [None, np.array([2.0, 2.1, 2.2])])
def test_variables_scale(
    enopt_config: Any,
    evaluator: Any,
    offsets: Optional[NDArray[np.float64]],
    scales: Optional[NDArray[np.float64]],
) -> None:
    initial_values = np.array(enopt_config["variables"]["initial_values"])
    lower_bounds = np.array([-2.0, -np.inf, -3.0])
    upper_bounds = np.array([np.inf, 1.0, 4.0])

    enopt_config["optimizer"]["max_iterations"] = 20
    enopt_config["variables"]["lower_bounds"] = lower_bounds
    enopt_config["variables"]["upper_bounds"] = upper_bounds

    if offsets is not None:
        enopt_config["variables"]["offsets"] = offsets
    if scales is not None:
        enopt_config["variables"]["scales"] = scales

    results = OptimizationPlanRunner(enopt_config, evaluator()).run().results
    assert results is not None

    if offsets is not None:
        initial_values = initial_values - offsets
        lower_bounds = lower_bounds - offsets
        upper_bounds = upper_bounds - offsets
    if scales is not None:
        initial_values = initial_values / scales
        lower_bounds = lower_bounds / scales
        upper_bounds = upper_bounds / scales
    config = EnOptConfig.model_validate(enopt_config)
    assert np.allclose(config.variables.initial_values, initial_values)
    assert np.allclose(config.variables.lower_bounds, lower_bounds)
    assert np.allclose(config.variables.upper_bounds, upper_bounds)
    result = results.evaluations.variables
    if scales is not None:
        result = result * scales
    if offsets is not None:
        result = result + offsets
    assert np.allclose(result, [0.0, 0.0, 0.5], atol=0.05)
    variables = (
        results.evaluations.variables
        if offsets is None and scales is None
        else results.evaluations.unscaled_variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.05)


def test_variables_scale_linear_constraints(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "rhs_values": [1.0, 0.75],
        "types": [ConstraintType.EQ, ConstraintType.EQ],
    }

    offsets = np.array([1.0, 1.1, 1.2])
    scales = np.array([2.0, 2.1, 2.2])
    enopt_config["variables"]["offsets"] = offsets
    enopt_config["variables"]["scales"] = scales

    config = EnOptConfig.model_validate(enopt_config)
    assert config.linear_constraints is not None
    coefficients = config.linear_constraints.coefficients
    rhs_values = config.linear_constraints.rhs_values
    assert np.allclose(
        coefficients / scales, enopt_config["linear_constraints"]["coefficients"]
    )
    assert np.allclose(
        rhs_values
        + np.matmul(coefficients if scales is None else coefficients / scales, offsets),
        enopt_config["linear_constraints"]["rhs_values"],
    )

    results = OptimizationPlanRunner(enopt_config, evaluator()).run().results
    assert results is not None
    assert results.evaluations.unscaled_variables is not None
    assert np.allclose(
        results.evaluations.unscaled_variables, [0.25, 0.0, 0.75], atol=0.02
    )


def test_check_linear_constraints(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
        "rhs_values": [0.0, 1.0, -1.0],
        "types": [ConstraintType.EQ, ConstraintType.LE, ConstraintType.GE],
    }
    enopt_config["optimizer"]["max_functions"] = 1
    results = OptimizationPlanRunner(enopt_config, evaluator()).run().results
    assert results is not None

    enopt_config["linear_constraints"]["rhs_values"] = [1.0, -1.0, 1.0]
    results = OptimizationPlanRunner(enopt_config, evaluator()).run().results
    assert results is None


def test_check_nonlinear_constraints(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": [0.0, 0.0, 0.0],
        "types": [ConstraintType.EQ, ConstraintType.LE, ConstraintType.GE],
        "scales": 10.0,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0]),
        lambda variables, _: cast(NDArray[np.float64], variables[0]),
        lambda variables, _: cast(NDArray[np.float64], variables[0]),
    )

    enopt_config["optimizer"]["max_functions"] = 1

    results = (
        OptimizationPlanRunner(enopt_config, evaluator(test_functions)).run().results
    )
    assert results is not None

    enopt_config["nonlinear_constraints"]["rhs_values"] = [1.0, -1.0, 1.0]

    results = (
        OptimizationPlanRunner(enopt_config, evaluator(test_functions)).run().results
    )
    assert results is None


def test_optimizer_variables_subset(enopt_config: Any, evaluator: Any) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    enopt_config["variables"]["initial_values"] = [0.0, 1.0, 0.1]
    enopt_config["variables"]["indices"] = [0, 2]

    def assert_gradient(event: Event) -> None:
        assert event.results
        assert event.results is not None
        for item in event.results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.weighted_objective[1] == 0.0
                assert np.all(item.gradients.objectives[:, 1] == 0.0)

    variables = (
        OptimizationPlanRunner(enopt_config, evaluator())
        .add_observer(EventType.FINISHED_EVALUATION, assert_gradient)
        .run()
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 1.0, 0.5], atol=0.02)


def test_optimizer_variables_subset_linear_constraints(
    enopt_config: Any, evaluator: Any
) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    enopt_config["variables"]["initial_values"] = [0.0, 1.0, 0.1]
    # The second and third constraints are dropped because they involve
    # variables that are not optimized.
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "rhs_values": [1.0, 1.0, 2.0],
        "types": [ConstraintType.EQ, ConstraintType.EQ, ConstraintType.EQ],
    }
    enopt_config["variables"]["indices"] = [0, 2]

    variables = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 1.0, 0.75], atol=0.02)


def test_parallelize(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"] = {
        "method": "differential_evolution",
        "max_iterations": 15,
        "options": {"seed": 123, "tol": 1e-10},
    }
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    enopt_config["variables"]["initial_values"][0] = 0.2

    enopt_config["optimizer"]["parallel"] = False
    variables = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)

    enopt_config["optimizer"]["parallel"] = True
    variables = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)


def test_rng(enopt_config: Any, evaluator: Any) -> None:
    variables1 = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables1 is not None
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)

    variables2 = (
        OptimizationPlanRunner(enopt_config, evaluator(), seed=DEFAULT_SEED)
        .run()
        .variables
    )
    assert variables2 is not None
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)
    assert np.all(variables1 == variables2)

    variables3 = (
        OptimizationPlanRunner(enopt_config, evaluator(), seed=DEFAULT_SEED + 123)
        .run()
        .variables
    )
    assert variables3 is not None
    assert np.allclose(variables3, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.all(variables3 == variables1)
