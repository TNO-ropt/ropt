from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.config.enopt import EnOptConfig
from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.enums import OptimizerExitCode
from ropt.plan import BasicOptimizer
from ropt.results import FunctionResults, GradientResults, Results
from ropt.transforms import OptModelTransforms, VariableScaler
from ropt.transforms.base import NonLinearConstraintTransform, ObjectiveTransform

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_max_functions_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 2
    enopt_config["optimizer"]["max_functions"] = max_functions
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run()
    assert last_evaluation == max_functions + 1
    assert optimizer.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED


def test_max_batches_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_batches = 2
    enopt_config["optimizer"]["max_batches"] = max_batches
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run()
    assert last_evaluation == max_batches
    assert optimizer.exit_code == OptimizerExitCode.MAX_BATCHES_REACHED


def test_max_functions_not_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 100
    enopt_config["optimizer"]["max_functions"] = max_functions
    enopt_config["optimizer"]["split_evaluations"] = True
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run()
    assert last_evaluation + 1 < 2 * max_functions
    assert optimizer.exit_code == OptimizerExitCode.OPTIMIZER_STEP_FINISHED


def test_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    def _observer(results: tuple[Results, ...]) -> None:
        assert isinstance(results[0], FunctionResults)
        assert results[0].functions is None

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(enopt_config, evaluator(functions)).set_results_callback(
        _observer
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_all_failed_realizations_not_supported(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["realizations"] = {"realization_min_success": 0}

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(
        enopt_config,
        evaluator(functions),
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_user_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _abort() -> bool:
        nonlocal last_evaluation

        if last_evaluation == 1:
            return True
        last_evaluation += 1
        return False

    optimizer = BasicOptimizer(enopt_config, evaluator()).set_abort_callback(_abort)
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
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


class ObjectiveScaler(ObjectiveTransform):
    def __init__(self, scales: NDArray[np.float64]) -> None:
        self._scales = scales

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives / self._scales

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scales


def test_objective_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    checked = False

    def check_value(results: tuple[Results, ...], value: float) -> None:
        nonlocal checked
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], value)

    config = EnOptConfig.model_validate(enopt_config)

    results1 = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results1 is not None
    assert results1.functions is not None
    variables1 = results1.evaluations.variables
    objectives1 = results1.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    init1 = test_functions[1](config.variables.initial_values, None)
    transforms = OptModelTransforms(
        objectives=ObjectiveScaler(np.array([init1, init1]))
    )

    optimizer = BasicOptimizer(
        EnOptConfig.model_validate(enopt_config, context=transforms),
        evaluator(),
        transforms=transforms,
    ).set_results_callback(partial(check_value, value=init1))
    results2 = optimizer.run().results
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, variables1, atol=0.02)
    assert results2.functions is not None
    assert np.allclose(objectives1, results2.functions.objectives, atol=0.025)


class ConstraintScaler(NonLinearConstraintTransform):
    def __init__(self, scales: NDArray[np.float64]) -> None:
        self._scales = scales

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_bounds / self._scales, upper_bounds / self._scales

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints / self._scales

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints * self._scales

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_diffs * self._scales, upper_diffs * self._scales


def test_constraint_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.4,
        "upper_bounds": np.inf,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    scales = np.array(
        test_functions[-1](enopt_config["variables"]["initial_values"], None), ndmin=1
    )
    transforms = OptModelTransforms(nonlinear_constraints=ConstraintScaler(scales))
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.lower_bounds == 0.4 / scales

    check = True

    def check_constraints(
        results: tuple[Results, ...], values: NDArray[np.float64]
    ) -> None:
        nonlocal check
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, values)

    BasicOptimizer(
        config, evaluator(test_functions), transforms=transforms
    ).set_results_callback(partial(check_constraints, values=scales)).run()


@pytest.mark.parametrize("offsets", [None, np.array([1.0, 1.1, 1.2])])
@pytest.mark.parametrize("scales", [None, np.array([2.0, 2.1, 2.2])])
def test_variables_scale_with_scaler(
    enopt_config: Any,
    evaluator: Any,
    offsets: NDArray[np.float64] | None,
    scales: NDArray[np.float64] | None,
) -> None:
    initial_values = np.array(enopt_config["variables"]["initial_values"])
    lower_bounds = np.array([-2.0, -np.inf, -3.0])
    upper_bounds = np.array([np.inf, 1.0, 4.0])

    enopt_config["optimizer"]["max_iterations"] = 20
    enopt_config["variables"]["lower_bounds"] = lower_bounds
    enopt_config["variables"]["upper_bounds"] = upper_bounds

    transforms = OptModelTransforms(variables=VariableScaler(scales, offsets))
    results = (
        BasicOptimizer(
            EnOptConfig.model_validate(enopt_config, context=transforms),
            evaluator(),
            transforms=transforms,
        )
        .run()
        .results
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
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert np.allclose(config.variables.initial_values, initial_values)
    assert np.allclose(config.variables.lower_bounds, lower_bounds)
    assert np.allclose(config.variables.upper_bounds, upper_bounds)
    result = results.evaluations.variables
    assert np.allclose(result, [0.0, 0.0, 0.5], atol=0.05)


def test_variables_scale_linear_constraints_with_scaler(
    enopt_config: Any,
    evaluator: Any,
) -> None:
    coefficients = [[1, 0, 1], [0, 1, 1]]
    lower_bounds = [1.0, 0.75]
    upper_bounds = [1.0, 0.75]

    enopt_config["linear_constraints"] = {
        "coefficients": coefficients,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }

    offsets = np.array([1.0, 1.1, 1.2])
    scales = np.array([2.0, 2.1, 2.2])

    transforms = OptModelTransforms(variables=VariableScaler(scales, offsets))
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.linear_constraints is not None
    transformed_coefficients = coefficients * scales
    transformed_scales = np.max(np.abs(transformed_coefficients), axis=-1)
    assert np.allclose(
        config.linear_constraints.coefficients,
        transformed_coefficients / transformed_scales[:, np.newaxis],
    )
    offsets = np.matmul(coefficients, offsets)
    assert np.allclose(
        config.linear_constraints.lower_bounds,
        (lower_bounds - offsets) / transformed_scales,
    )
    assert np.allclose(
        config.linear_constraints.upper_bounds,
        (upper_bounds - offsets) / transformed_scales,
    )

    results = BasicOptimizer(config, evaluator(), transforms=transforms).run().results
    assert results is not None
    assert results.evaluations.variables is not None
    assert np.allclose(results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02)


def test_check_linear_constraints(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
        "lower_bounds": [0.0, -np.inf, -1.0],
        "upper_bounds": [0.0, 1.0, np.inf],
    }
    results1 = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results1 is not None

    enopt_config["linear_constraints"]["lower_bounds"] = [0.0, -np.inf, -1.0]
    enopt_config["linear_constraints"]["upper_bounds"] = [0.0, 1.0, np.inf]
    results2 = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results2 is not None
    assert np.allclose(results1.evaluations.variables, results2.evaluations.variables)

    enopt_config["linear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    enopt_config["linear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    results = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results is None


def test_check_nonlinear_constraints(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.0, -np.inf, 0.0],
        "upper_bounds": [0.0, 0.0, np.inf],
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0],
        lambda variables, _: variables[0],
        lambda variables, _: variables[0],
    )

    results1 = BasicOptimizer(enopt_config, evaluator(test_functions)).run().results
    assert results1 is not None

    # Flipping the bounds should still work:
    enopt_config["nonlinear_constraints"]["lower_bounds"] = [0.0, -np.inf, 0.0]
    enopt_config["nonlinear_constraints"]["upper_bounds"] = [0.0, 0.0, np.inf]
    results2 = BasicOptimizer(enopt_config, evaluator(test_functions)).run().results
    assert results2 is not None
    assert np.allclose(results1.evaluations.variables, results2.evaluations.variables)

    enopt_config["nonlinear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    enopt_config["nonlinear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    results = BasicOptimizer(enopt_config, evaluator(test_functions)).run().results
    assert results is None


def test_optimizer_variables_subset(enopt_config: Any, evaluator: Any) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    enopt_config["variables"]["initial_values"] = [0.0, 1.0, 0.1]
    enopt_config["variables"]["mask"] = [True, False, True]

    def assert_gradient(results: tuple[Results, ...]) -> None:
        for item in results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.weighted_objective[1] == 0.0
                assert np.all(item.gradients.objectives[:, 1] == 0.0)

    variables = (
        BasicOptimizer(enopt_config, evaluator())
        .set_results_callback(assert_gradient)
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
        "lower_bounds": [1.0, 1.0, 2.0],
        "upper_bounds": [1.0, 1.0, 2.0],
    }
    enopt_config["variables"]["mask"] = [True, False, True]

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
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
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)

    enopt_config["optimizer"]["parallel"] = True
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)


def test_rng(enopt_config: Any, evaluator: Any) -> None:
    variables1 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables1 is not None
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)

    variables2 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables2 is not None
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)
    assert np.all(variables1 == variables2)

    enopt_config["gradient"]["seed"] = (1, DEFAULT_SEED)
    variables3 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables3 is not None
    assert np.allclose(variables3, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.all(variables3 == variables1)
