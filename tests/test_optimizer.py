from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.config import EnOptConfig
from ropt.config.constants import DEFAULT_SEED
from ropt.enums import EventType, ExitCode
from ropt.plan import BasicOptimizer
from ropt.plugins._manager import PluginManager
from ropt.results import FunctionResults, GradientResults, Results
from ropt.transforms import OptModelTransforms, VariableScaler
from ropt.transforms.base import NonLinearConstraintTransform, ObjectiveTransform

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ropt.plan import Event

_SLSQP = "slsqp"
_DIFFERENTIAL_EVOLUTION = "differential_evolution"

pytestmark = [
    pytest.mark.parametrize(
        "external", ["", pytest.param("external/", marks=pytest.mark.external)]
    )
]

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "method": _SLSQP,
            "max_iterations": 15,
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_basic_run(enopt_config: Any, evaluator: Any, external: str) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [0, 0, 0.5], atol=0.02)


def test_invalid_options(enopt_config: Any, external: str) -> None:
    enopt_config["optimizer"]["options"] = {"ftol": 0.1, "foo": 1}
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    method = enopt_config["optimizer"]["method"]
    plugin = PluginManager().get_plugin("optimizer", method)
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`"
    ):
        plugin.validate_options(method, enopt_config["optimizer"]["options"])


def test_max_functions_exceeded(
    enopt_config: Any, evaluator: Any, external: str
) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 2
    enopt_config["optimizer"]["max_functions"] = max_functions
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run(initial_values)
    assert last_evaluation == max_functions + 1
    assert optimizer.exit_code == ExitCode.MAX_FUNCTIONS_REACHED


def test_max_batches_exceeded(enopt_config: Any, evaluator: Any, external: str) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_batches = 2
    enopt_config["optimizer"]["max_batches"] = max_batches
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run(initial_values)
    assert last_evaluation == max_batches
    assert optimizer.exit_code == ExitCode.MAX_BATCHES_REACHED


def test_max_functions_not_exceeded(
    enopt_config: Any, evaluator: Any, external: str
) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 100
    enopt_config["optimizer"]["max_functions"] = max_functions
    enopt_config["gradient"] = {"evaluation_policy": "separate"}
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run(initial_values)
    assert last_evaluation + 1 < 2 * max_functions
    assert optimizer.exit_code == ExitCode.OPTIMIZER_STEP_FINISHED


def test_failed_realizations(enopt_config: Any, evaluator: Any, external: str) -> None:
    def _observer(results: tuple[Results, ...]) -> None:
        assert isinstance(results[0], FunctionResults)
        assert results[0].functions is None

    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(enopt_config, evaluator(functions)).set_results_callback(
        _observer
    )
    optimizer.run(initial_values)
    assert optimizer.exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_failed_realizations_constraints(
    enopt_config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    def _observer(results: tuple[Results, ...]) -> None:
        assert isinstance(results[0], FunctionResults)
        assert results[0].functions is None

    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    functions = (*test_functions, lambda _0, _1: np.array(np.nan))

    optimizer = BasicOptimizer(enopt_config, evaluator(functions)).set_results_callback(
        _observer
    )
    optimizer.run(initial_values)
    assert optimizer.exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_all_failed_realizations_not_supported(
    enopt_config: Any, evaluator: Any, external: str
) -> None:
    enopt_config["realizations"] = {"realization_min_success": 0}

    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(enopt_config, evaluator(functions))
    optimizer.run(initial_values)
    assert optimizer.exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_user_abort(enopt_config: Any, evaluator: Any, external: str) -> None:
    last_evaluation = 0

    def _abort() -> bool:
        nonlocal last_evaluation

        if last_evaluation == 1:
            return True
        last_evaluation += 1
        return False

    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    optimizer = BasicOptimizer(enopt_config, evaluator()).set_abort_callback(_abort)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert last_evaluation == 1
    assert optimizer.exit_code == ExitCode.USER_ABORT


def test_single_perturbation(enopt_config: Any, evaluator: Any, external: str) -> None:
    enopt_config["gradient"] = {
        "number_of_perturbations": 1,
        "merge_realizations": True,
    }

    enopt_config["realizations"] = {"weights": 5 * [1]}
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_external_error(enopt_config: Any, evaluator: Any, external: str) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    enopt_config["optimizer"]["options"] = {"ftol": "foo"}
    err = "Input should be a valid number, unable to parse string as a number"
    with pytest.raises(ValueError, match=err):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


class ObjectiveScaler(ObjectiveTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives / self._scales

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scales


def test_objective_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    results1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results1 is not None
    assert results1.functions is not None
    variables1 = results1.evaluations.variables
    objectives1 = results1.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    def function1(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[0](variables, None))

    def function2(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[1](variables, None))

    init1 = test_functions[1](initial_values, None)
    transforms = OptModelTransforms(
        objectives=ObjectiveScaler(np.array([init1, init1]))
    )

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(event.data["transforms"])
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, variables1, atol=0.02)
    assert results2.functions is not None
    assert np.allclose(objectives1, results2.functions.objectives, atol=0.025)


def test_objective_with_lazy_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    results1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results1 is not None
    assert results1.functions is not None
    variables1 = results1.evaluations.variables
    objectives1 = results1.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    objective_transform = ObjectiveScaler(np.array([1.0, 1.0]))
    transforms = OptModelTransforms(objectives=objective_transform)

    init1 = test_functions[1](initial_values, None)

    def function1(variables: NDArray[np.float64], _: Any) -> float:
        objective_transform.set_scales([init1, init1])
        return float(test_functions[0](variables, None))

    def function2(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[1](variables, None))

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(event.data["transforms"])
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, variables1, atol=0.02)
    assert results2.functions is not None
    assert np.allclose(objectives1, results2.functions.objectives, atol=0.025)


class ConstraintScaler(NonLinearConstraintTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

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


def test_nonlinear_constraint_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    results1 = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).results
    )
    assert results1 is not None
    assert results1.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert results1.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scales = np.array(functions[-1](initial_values, None), ndmin=1)
    transforms = OptModelTransforms(nonlinear_constraints=ConstraintScaler(scales))
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds[1] == 0.4 / scales

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(event.data["transforms"])
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(
        results2.evaluations.variables, results1.evaluations.variables, atol=0.02
    )
    assert results1.functions is not None
    assert results2.functions is not None
    assert np.allclose(
        results1.functions.objectives, results2.functions.objectives, atol=0.025
    )


def test_nonlinear_constraint_with_lazy_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    results1 = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).results
    )
    assert results1 is not None
    assert results1.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert results1.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scales = np.array(functions[-1](initial_values, None), ndmin=1)
    scaler = ConstraintScaler([1.0])
    transforms = OptModelTransforms(nonlinear_constraints=scaler)

    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds[1] == 0.4

    def constraint_function(variables: NDArray[np.float64], _: Any) -> float:
        scaler.set_scales(scales)
        return float(variables[0] + variables[2])

    functions = (*test_functions, constraint_function)

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        results = event.data.get("results", ())
        config = event.data["config"]
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert config.nonlinear_constraints is not None
                assert transforms.nonlinear_constraints is not None
                _, upper_bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
                    config.nonlinear_constraints.lower_bounds,
                    config.nonlinear_constraints.upper_bounds,
                )
                assert np.allclose(upper_bounds, 0.4 / scales)
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(event.data["transforms"])
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(
        results2.evaluations.variables, results1.evaluations.variables, atol=0.02
    )
    assert results1.functions is not None
    assert results2.functions is not None
    assert np.allclose(
        results1.functions.objectives, results2.functions.objectives, atol=0.025
    )


@pytest.mark.parametrize("offsets", [None, np.array([1.0, 1.1, 1.2])])
@pytest.mark.parametrize("scales", [None, np.array([2.0, 2.1, 2.2])])
def test_variables_scale_with_scaler(
    enopt_config: Any,
    evaluator: Any,
    offsets: NDArray[np.float64] | None,
    scales: NDArray[np.float64] | None,
    external: str,
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

    lower_bounds = np.array([-2.0, -np.inf, -3.0])
    upper_bounds = np.array([np.inf, 1.0, 4.0])

    enopt_config["optimizer"]["max_iterations"] = 20
    enopt_config["variables"]["lower_bounds"] = lower_bounds
    enopt_config["variables"]["upper_bounds"] = upper_bounds

    transforms = OptModelTransforms(variables=VariableScaler(scales, offsets))
    results = (
        BasicOptimizer(enopt_config, evaluator(), transforms=transforms)
        .run(initial_values)
        .results
    )
    assert results is not None

    if offsets is not None:
        lower_bounds = lower_bounds - offsets
        upper_bounds = upper_bounds - offsets
    if scales is not None:
        lower_bounds = lower_bounds / scales
        upper_bounds = upper_bounds / scales
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert np.allclose(config.variables.lower_bounds, lower_bounds)
    assert np.allclose(config.variables.upper_bounds, upper_bounds)
    result = results.evaluations.variables
    assert np.allclose(result, [0.0, 0.0, 0.5], atol=0.05)


def test_variables_scale_linear_constraints_with_scaler(
    enopt_config: Any, evaluator: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"

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

    results = (
        BasicOptimizer(enopt_config, evaluator(), transforms=transforms)
        .run(initial_values)
        .results
    )
    assert results is not None
    assert results.evaluations.variables is not None
    assert np.allclose(results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02)


def test_check_linear_constraints(
    enopt_config: Any, evaluator: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
        "lower_bounds": [0.0, -np.inf, -1.0],
        "upper_bounds": [0.0, 1.0, np.inf],
    }
    results1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results1 is not None

    enopt_config["linear_constraints"]["lower_bounds"] = [0.0, -np.inf, -1.0]
    enopt_config["linear_constraints"]["upper_bounds"] = [0.0, 1.0, np.inf]
    results2 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results2 is not None
    assert np.allclose(results1.evaluations.variables, results2.evaluations.variables)

    enopt_config["linear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    enopt_config["linear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    results = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results is None


def test_check_nonlinear_constraints(
    enopt_config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
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

    results1 = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .results
    )
    assert results1 is not None

    # Flipping the bounds should still work:
    enopt_config["nonlinear_constraints"]["lower_bounds"] = [0.0, -np.inf, 0.0]
    enopt_config["nonlinear_constraints"]["upper_bounds"] = [0.0, 0.0, np.inf]
    results2 = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .results
    )
    assert results2 is not None
    assert np.allclose(results1.evaluations.variables, results2.evaluations.variables)

    enopt_config["nonlinear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    enopt_config["nonlinear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    results = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .results
    )
    assert results is None


def test_optimizer_variables_subset(
    enopt_config: Any, evaluator: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
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
        .run([0.0, 1.0, 0.1])
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 1.0, 0.5], atol=0.02)


def test_optimizer_variables_subset_linear_constraints(
    enopt_config: Any, evaluator: Any, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem. The
    # second and third constraints are dropped because they involve variables
    # that are not optimized.
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "lower_bounds": [1.0, 1.0, 2.0],
        "upper_bounds": [1.0, 1.0, 2.0],
    }
    enopt_config["variables"]["mask"] = [True, False, True]

    variables = BasicOptimizer(enopt_config, evaluator()).run([0.0, 1.0, 0.1]).variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 1.0, 0.75], atol=0.02)


def test_parallelize(enopt_config: Any, evaluator: Any, external: str) -> None:
    enopt_config["optimizer"] = {
        "method": f"{external}{_DIFFERENTIAL_EVOLUTION}",
        "max_iterations": 15,
        "options": {"rng": 123, "tol": 1e-10},
    }
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]

    enopt_config["optimizer"]["parallel"] = False
    variables = (
        BasicOptimizer(enopt_config, evaluator())
        .run([0.2, *initial_values[1:]])
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)

    enopt_config["optimizer"]["parallel"] = True
    variables = (
        BasicOptimizer(enopt_config, evaluator())
        .run([0.2, *initial_values[1:]])
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)


def test_rng(enopt_config: Any, evaluator: Any, external: str) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    variables1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables1 is not None
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)

    variables2 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables2 is not None
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)
    assert np.all(variables1 == variables2)

    enopt_config["variables"]["seed"] = (1, DEFAULT_SEED)
    variables3 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables3 is not None
    assert np.allclose(variables3, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.all(variables3 == variables1)


def test_arbitrary_objective_weights(
    enopt_config: Any, evaluator: Any, external: str, test_functions: Any
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}{_SLSQP}"
    new_functions = (
        *test_functions,
        lambda variables, _: test_functions[1](variables, None),
    )

    enopt_config["objectives"]["weights"] = [0.75, 0.25, -0.25]
    variables = (
        BasicOptimizer(enopt_config, evaluator(new_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert not np.allclose(variables, [0, 0, 0.5], atol=0.02)

    enopt_config["objectives"]["weights"] = [0.75, 0.25, 0.0]
    variables = (
        BasicOptimizer(enopt_config, evaluator(new_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0, 0, 0.5], atol=0.02)

    enopt_config["objectives"]["weights"] = [0.75, 0.5, -0.25]
    variables = (
        BasicOptimizer(enopt_config, evaluator(new_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0, 0, 0.5], atol=0.02)

    enopt_config["objectives"]["weights"] = [-0.75, -0.25]
    with pytest.raises(ValidationError, match="The sum of weights is not positive"):
        EnOptConfig.model_validate(enopt_config)
