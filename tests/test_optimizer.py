from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.config.plan import StepConfig
from ropt.enums import ConstraintType, EventType, OptimizerExitCode
from ropt.events import OptimizationEvent
from ropt.exceptions import ConfigError
from ropt.optimization import EnsembleOptimizer, Plan, PlanContext
from ropt.plugins import PluginManager
from ropt.plugins.optimization_steps.evaluator import DefaultEvaluatorStep
from ropt.report import ResultsDataFrame
from ropt.results import FunctionResults, GradientResults


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "optimizer": {
            "algorithm": "slsqp",
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


def test_no_enopt_set(evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(ConfigError, match="Optimizer configuration missing"):
        optimizer.start_optimization(plan=[{"optimizer": {}}])


def test_basic_optimization_step(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_duplicate_tracker(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(ConfigError, match="Duplicate step ID: optimum"):
        optimizer.start_optimization(
            plan=[
                {"config": enopt_config},
                {"tracker": {"id": "optimum", "source": "opt"}},
                {"optimizer": {"id": "opt"}},
                {"tracker": {"id": "optimum", "source": "opt"}},
            ],
        )


def test_max_functions_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def track_results(event: OptimizationEvent) -> None:
        nonlocal last_evaluation
        assert event.results is not None
        last_evaluation = event.results[0].result_id

    def handle_finished(event: OptimizationEvent) -> None:
        assert event.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED

    max_functions = 2
    enopt_config["optimizer"]["max_functions"] = max_functions
    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, track_results)
    optimizer.add_observer(EventType.FINISHED_OPTIMIZER_STEP, handle_finished)
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])
    assert last_evaluation == max_functions


def test_max_functions_not_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def track_results(event: OptimizationEvent) -> None:
        nonlocal last_evaluation
        assert event.results is not None
        last_evaluation = event.results[0].result_id

    def handle_finished(event: OptimizationEvent) -> None:
        assert event.exit_code == OptimizerExitCode.OPTIMIZER_STEP_FINISHED

    max_functions = 100
    enopt_config["optimizer"]["max_functions"] = max_functions
    enopt_config["optimizer"]["split_evaluations"] = True

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, track_results)
    optimizer.add_observer(EventType.FINISHED_OPTIMIZER_STEP, handle_finished)
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])
    assert last_evaluation + 1 < 2 * max_functions


def test_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    def _observer(event: OptimizationEvent) -> None:
        assert event.results is not None
        assert isinstance(event.results[0], FunctionResults)
        assert event.results[0].functions is None

    def handle_finished(event: OptimizationEvent) -> None:
        assert event.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = EnsembleOptimizer(evaluator(functions))
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _observer)
    optimizer.add_observer(EventType.FINISHED_OPTIMIZER_STEP, handle_finished)
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])


def test_user_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def _observer(event: OptimizationEvent, optimizer: EnsembleOptimizer) -> None:
        nonlocal last_evaluation
        assert event.results is not None
        last_evaluation = event.results[0].result_id
        if event.results[0].result_id == 1:
            optimizer.abort_optimization()

    def handle_finished(event: OptimizationEvent) -> None:
        assert event.exit_code == OptimizerExitCode.USER_ABORT

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION, partial(_observer, optimizer=optimizer)
    )
    optimizer.add_observer(EventType.FINISHED_OPTIMIZER_STEP, handle_finished)
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert last_evaluation == 1


def test_single_perturbation(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["gradient"] = {
        "number_of_perturbations": 1,
        "merge_realizations": True,
    }
    enopt_config["realizations"] = {"weights": 5 * [1]}

    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_objective_auto_scale(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    init1 = test_functions[1](config.variables.initial_values, None)

    enopt_config["objective_functions"]["scales"] = [1.0, init1]
    enopt_config["objective_functions"]["auto_scale"] = False

    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    manual_result = results.evaluations.variables

    enopt_config["objective_functions"]["scales"] = [1.0, 1.0]
    enopt_config["objective_functions"]["auto_scale"] = [False, True]

    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, manual_result)

    enopt_config["objective_functions"]["scales"] = [1.0, 2.0 * init1]
    enopt_config["objective_functions"]["auto_scale"] = False

    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    manual_result = results.evaluations.variables

    enopt_config["objective_functions"]["scales"] = [1.0, 2.0]
    enopt_config["objective_functions"]["auto_scale"] = [False, True]

    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, manual_result)


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

    def check_constraints(event: OptimizationEvent) -> None:
        assert event.results is not None
        for item in event.results:
            if isinstance(item, FunctionResults) and item.result_id == 0:
                assert item.functions is not None
                assert item.functions.scaled_constraints is not None
                assert np.allclose(item.functions.scaled_constraints, 1.0)

    enopt_config["nonlinear_constraints"]["scales"] = scales
    enopt_config["nonlinear_constraints"]["auto_scale"] = False

    optimizer = EnsembleOptimizer(evaluator(test_functions))
    optimizer.add_observer(EventType.FINISHED_EVALUATION, check_constraints)
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])

    enopt_config["nonlinear_constraints"]["scales"] = 1.0
    enopt_config["nonlinear_constraints"]["auto_scale"] = True

    optimizer = EnsembleOptimizer(evaluator(test_functions))
    optimizer.add_observer(EventType.FINISHED_EVALUATION, check_constraints)
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])


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

    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                },
            },
        ],
    )
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

    optimizer = EnsembleOptimizer(evaluator())

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

    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                },
            },
        ],
    )
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
    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                },
            },
        ],
    )
    assert results is not None

    enopt_config["linear_constraints"]["rhs_values"] = [1.0, -1.0, 1.0]
    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                },
            },
        ],
    )
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

    optimizer = EnsembleOptimizer(evaluator(test_functions))
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                },
            },
        ],
    )
    assert results is not None

    enopt_config["nonlinear_constraints"]["rhs_values"] = [1.0, -1.0, 1.0]

    optimizer = EnsembleOptimizer(evaluator(test_functions))
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                },
            },
        ],
    )
    assert results is None


def test_optimizer_variables_subset(enopt_config: Any, evaluator: Any) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    enopt_config["variables"]["initial_values"] = [0.0, 1.0, 0.1]
    enopt_config["variables"]["indices"] = [0, 2]

    def assert_gradient(event: OptimizationEvent) -> None:
        assert event.results is not None
        for item in event.results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.weighted_objective[1] == 0.0
                assert np.all(np.equal(item.gradients.objectives[:, 1], 0.0))

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, assert_gradient)
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 1.0, 0.5], atol=0.02)


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

    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                },
            },
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.25, 1.0, 0.75], atol=0.02)


def test_optimization_sequential(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 2

    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt1"}},
            {"tracker": {"id": "last", "source": "opt1", "type": "last_result"}},
            {"config": enopt_config2},
            {"update_config": {"initial_variables": "last"}},
            {"optimizer": {}},
        ],
    )
    assert not np.allclose(
        completed[1].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert np.all(
        completed[2].evaluations.variables == completed[1].evaluations.variables
    )


def test_two_optimizers_alternating(enopt_config: Any, evaluator: Any) -> None:
    completed_functions = 0

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed_functions
        for item in event.results:
            if isinstance(item, FunctionResults):
                completed_functions += 1

    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    opt_config1 = {
        "speculative": True,
        "max_functions": 4,
    }
    opt_config2 = {
        "algorithm": "slsqp",
        "speculative": True,
        "max_functions": 3,
    }

    enopt_config1 = deepcopy(enopt_config)
    enopt_config1["variables"]["indices"] = [0, 2]
    enopt_config1["optimizer"] = opt_config1
    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["variables"]["indices"] = [1]
    enopt_config2["optimizer"] = opt_config2

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    results = optimizer.start_optimization(
        plan=[
            {
                "tracker": {
                    "id": "last",
                    "source": {"opt1", "opt2", "opt3"},
                    "type": "last_result",
                }
            },
            {
                "tracker": {
                    "id": "optimum",
                    "source": {"opt1", "opt2", "opt3", "opt4"},
                },
            },
            {"config": enopt_config1},
            {"optimizer": {"id": "opt1"}},
            {"config": enopt_config2},
            {"update_config": {"initial_variables": "last"}},
            {"optimizer": {"id": "opt2"}},
            {"config": enopt_config1},
            {"update_config": {"initial_variables": "last"}},
            {"optimizer": {"id": "opt3"}},
            {"config": enopt_config2},
            {"update_config": {"initial_variables": "last"}},
            {"optimizer": {"id": "opt4"}},
        ],
    )

    assert completed_functions == 14
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_two_optimizers_nested(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    completed_functions = 0

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed_functions
        for item in event.results:
            if isinstance(item, FunctionResults):
                completed_functions += 1

    enopt_config["optimizer"]["tolerance"] = 1e-10
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4
    enopt_config["variables"]["indices"] = [0, 2]
    nested_config = deepcopy(enopt_config)
    nested_config["variables"]["indices"] = [1]
    enopt_config["optimizer"]["max_functions"] = 5

    inner_steps = [
        {"config": nested_config},
        {"optimizer": {"id": "nested_optimizer"}},
        {"tracker": {"id": "nested_optimum", "source": "nested_optimizer"}},
    ]
    outer_steps = [
        {"config": enopt_config},
        {
            "optimizer": {
                "id": "opt",
                "nested_plan": inner_steps,
            },
        },
        {"tracker": {"id": "optimum", "source": "opt"}},
    ]

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    results = optimizer.start_optimization(plan=outer_steps)

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert completed_functions == 25


def test_parallelize(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"] = {
        "algorithm": "differential_evolution",
        "max_iterations": 10,
        "options": {"seed": 123, "tol": 1e-10},
    }
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    enopt_config["variables"]["initial_values"][0] = 0.2

    enopt_config["optimizer"]["parallel"] = False
    optimizer = EnsembleOptimizer(evaluator())
    optimum1 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert optimum1 is not None

    enopt_config["optimizer"]["parallel"] = True
    optimizer = EnsembleOptimizer(evaluator())
    optimum2 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert optimum2 is not None

    assert np.allclose(optimum1.evaluations.variables, [0.15, 0.0, 0.2], atol=3e-2)
    assert np.allclose(optimum2.evaluations.variables, [0.15, 0.0, 0.2], atol=3e-2)


def test_rng(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())

    result1 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result1 is not None
    assert np.allclose(result1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    result2 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
        seed=DEFAULT_SEED,
    )
    assert result2 is not None
    assert np.all(
        np.equal(result1.evaluations.variables, result2.evaluations.variables)
    )

    result3 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
        seed=DEFAULT_SEED + 1,
    )
    assert result3 is not None
    assert not np.all(
        np.equal(result1.evaluations.variables, result3.evaluations.variables)
    )


def test_restart_step(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4
    optimizer = EnsembleOptimizer(evaluator())
    results1 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results1 is not None
    assert np.allclose(results1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    optimizer = EnsembleOptimizer(evaluator())
    results2 = optimizer.start_optimization(
        plan=[
            {"tracker": {"id": "optimum", "source": "opt"}},
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"restart": {"max_restarts": 0}},
        ],
    )
    assert results2 is not None
    assert np.all(results1.evaluations.variables == results2.evaluations.variables)


def test_restart_initial(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1}},
        ],
    )

    assert len(completed) == 6

    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)


def test_restart_not_at_the_end(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1}},
            {"optimizer": {}},
        ],
    )

    assert len(completed) == 9

    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)
    assert np.all(completed[6].evaluations.variables == initial)


def test_restart_label(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"label": "restart_here"},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1, "label": "restart_here"}},
        ],
    )

    assert len(completed) == 9


def test_restart_last(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True

    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"update_config": {"initial_variables": "last"}},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"source": "opt", "type": "last_result", "id": "last"}},
            {"restart": {"max_restarts": 1}},
        ],
    )

    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )


def test_restart_optimal(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"tracker": {"id": "optimum", "source": "opt"}},
            {"config": enopt_config},
            {"update_config": {"initial_variables": "optimum"}},
            {"optimizer": {"id": "opt"}},
            {"restart": {"max_restarts": 1}},
        ],
    )

    # The third evaluation is the optimum, and used to restart:
    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )


def test_restart_optimal_step(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    completed: List[FunctionResults] = []
    max_functions = 5

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    # Make sure each restart has worse objectives, and that the last evaluation
    # is even worse, so each run has its own optimum that is worse than the
    # global and not at its last evaluation. We should end up with initial
    # values that are not from the global optimum, or from the most recent
    # evaluation:
    new_functions = (
        lambda variables, context: (
            test_functions[0](variables, context)
            + int((len(completed) + 1) / max_functions)
        ),
        lambda variables, context: (
            test_functions[1](variables, context)
            + int((len(completed) + 1) / max_functions)
        ),
    )

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = max_functions

    optimizer = EnsembleOptimizer(evaluator(new_functions))
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"tracker": {"id": "optimum", "source": "opt"}},
            {"update_config": {"initial_variables": "optimum"}},
            {"optimizer": {"id": "opt"}},
            {"reset_tracker": "optimum"},
            {"restart": {"max_restarts": 2}},
        ],
    )

    # The third evaluation is the optimum, and used to restart the second run:
    assert np.all(
        completed[max_functions].evaluations.variables
        == completed[2].evaluations.variables
    )
    # The 8th evaluation is the optimum of the second run, and used for the third:
    assert np.all(
        completed[2 * max_functions].evaluations.variables
        == completed[8].evaluations.variables
    )


def test_restart_metadata(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(
        {"result_id", "evaluations.variables", "metadata.restart"}
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        lambda event: reporter.add_results(event.config, event.results),  # type: ignore
    )
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1}},
        ]
    )

    assert reporter.frame["metadata.restart"].to_list() == 3 * [0] + 3 * [1]


class ModifyConfig:
    def __init__(
        self,
        _0: StepConfig,
        _1: PlanContext,
        plan: Plan,
        weights: NDArray[np.float64],
    ) -> None:
        self._plan = plan
        self._weights = weights

    def run(self, _: Optional[NDArray[np.float64]]) -> bool:
        self._plan.update_enopt_config(
            {"objective_functions": {"weights": self._weights}}
        )
        return False


def test_modify_enopt_in_plan(enopt_config: Any, evaluator: Any) -> None:
    weights = enopt_config["objective_functions"]["weights"]
    enopt_config["objective_functions"]["weights"] = [1, 1]
    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert not np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    plugin_manager = PluginManager()
    plugin_manager.add_backends(
        "optimization_step",
        {
            "modify_backend": lambda config, context, plan: ModifyConfig(
                config, context, plan, weights
            )
        },
    )

    optimizer = EnsembleOptimizer(evaluator(), plugin_manager=plugin_manager)
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"backend": "modify_backend", "type": "default"},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


class EvaluatorWithProcess(DefaultEvaluatorStep):
    def __init__(
        self,
        config: Dict[str, Any],
        context: PlanContext,
        plan: Plan,
        completed: List[FunctionResults],
    ) -> None:
        super().__init__(config, context, plan)
        self._completed = completed

    def process(self, results: FunctionResults) -> None:
        self._completed.append(results)


@pytest.mark.parametrize("backend", ["default", "evaluator_backend"])
@pytest.mark.parametrize("init_from", [None, "last", "optimum"])
def test_evaluator_step(
    enopt_config: Any, evaluator: Any, init_from: str, backend: str
) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    plugin_manager = PluginManager()
    if backend == "evaluator_backend":
        plugin_manager.add_backends(
            "optimization_step",
            {
                "evaluator_backend": lambda config, context, plan: EvaluatorWithProcess(
                    config.model_extra["evaluator"], context, plan, completed
                )
            },
        )

    optimizer = EnsembleOptimizer(evaluator(), plugin_manager=plugin_manager)
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    if backend == "default":
        optimizer.add_observer(EventType.FINISHED_EVALUATOR_STEP, _track_evaluations)

    plan = [
        {"config": enopt_config},
        {"optimizer": {"id": "opt"}},
    ]
    if init_from is None:
        plan.append({"backend": backend, "evaluator": {}})
    elif init_from == "last":
        plan.extend(
            [
                {"tracker": {"id": "last", "source": "opt", "type": "last_result"}},
                {"update_config": {"initial_variables": "last"}},
                {"backend": backend, "evaluator": {}},
            ]
        )
    else:
        plan.extend(
            [
                {"update_config": {"initial_variables": "optimum"}},
                {"backend": backend, "evaluator": {}},
            ]
        )
    plan.append({"tracker": {"id": "optimum", "source": "opt"}})
    results = optimizer.start_optimization(plan=plan)

    from_map = {None: 0, "last": 3, "optimum": 2}

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert results.result_id == 2 * from_map["optimum"]
    assert np.all(
        completed[from_map[init_from]].evaluations.variables
        == completed[-1].evaluations.variables
    )


def test_restart_last_optimum_from_evaluation(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["optimizer"]["speculative"] = True

    completed: List[Tuple[EventType, float]] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            (event.event_type, float(item.functions.weighted_objective))
            for item in event.results
            if isinstance(item, FunctionResults) and item.functions is not None
        ]

    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.add_observer(EventType.FINISHED_EVALUATOR_STEP, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"tracker": {"id": "eval_tracker", "source": "eval"}},
            {"tracker": {"id": "opt_tracker", "source": "opt"}},
            {"update_config": {"initial_variables": "eval_tracker"}},
            {"optimizer": {"id": "opt"}},
            {"update_config": {"initial_variables": "opt_tracker"}},
            {"evaluator": {"id": "eval"}},
            {"reset_tracker": "opt_tracker"},
            {"restart": {"max_restarts": 3}},
        ]
    )

    min_opt = min(
        item[1] for item in completed if item[0] == EventType.FINISHED_EVALUATION
    )
    min_eval = min(
        item[1] for item in completed if (item[0] == EventType.FINISHED_EVALUATOR_STEP)
    )
    assert optimizer.results["eval_tracker"] is not None
    assert optimizer.results["eval_tracker"].functions is not None
    assert optimizer.results["eval_tracker"].functions.weighted_objective == min_eval
    assert optimizer.results["opt_tracker"] is not None
    assert optimizer.results["opt_tracker"].functions is not None
    assert optimizer.results["opt_tracker"].functions.weighted_objective == min_eval

    assert min_opt == min_eval
