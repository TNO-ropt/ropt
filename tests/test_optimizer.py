# ruff: noqa: RUF069

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.config import (
    NonlinearConstraintTransformConfig,
    ObjectiveTransformConfig,
    VariableTransformConfig,
)
from ropt.config.constants import DEFAULT_SEED
from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, ExitCode
from ropt.results import FunctionResults, GradientResults, Results
from ropt.transforms.default import (
    DefaultNonlinearConstraintTransform,
    DefaultObjectiveTransform,
    DefaultVariableTransform,
)
from ropt.workflow import BasicOptimizer, validate_backend_options

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.events import EnOptEvent

_SLSQP = "slsqp"
_DIFFERENTIAL_EVOLUTION = "differential_evolution"

pytestmark = [
    pytest.mark.parametrize(
        "external", ["", pytest.param("external/", marks=pytest.mark.external)]
    )
]

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 20,
        },
        "backend": {
            "method": _SLSQP,
            "max_iterations": 15,
            "convergence_tolerance": 1e-5,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_basic_run(config: Any, evaluator: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(optimizer.results.evaluations.variables, [0, 0, 0.5], atol=0.02)


def test_invalid_options(config: Any, external: str) -> None:
    config["backend"]["options"] = {"ftol": 0.1, "foo": 1}
    config["backend"]["method"] = f"{external}{_SLSQP}"

    method = config["backend"]["method"]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`"
    ):
        validate_backend_options(method, config["backend"]["options"])


def test_max_functions_exceeded(config: Any, evaluator: Any, external: str) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 2
    config["optimizer"]["max_functions"] = max_functions
    config["backend"]["method"] = f"{external}{_SLSQP}"
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_results_callback(track_results)
    exit_code = optimizer.run(initial_values)
    assert last_evaluation == max_functions + 1
    assert exit_code == ExitCode.MAX_FUNCTIONS_REACHED


def test_max_batches_exceeded(config: Any, evaluator: Any, external: str) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_batches = 2
    config["optimizer"]["max_batches"] = max_batches
    config["backend"]["method"] = f"{external}{_SLSQP}"
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_results_callback(track_results)
    exit_code = optimizer.run(initial_values)
    assert last_evaluation == max_batches
    assert exit_code == ExitCode.MAX_BATCHES_REACHED


def test_max_functions_not_exceeded(config: Any, evaluator: Any, external: str) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 100
    config["optimizer"]["max_functions"] = max_functions
    config["gradient"] = {"evaluation_policy": "separate"}
    config["backend"]["method"] = f"{external}{_SLSQP}"
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_results_callback(track_results)
    exit_code = optimizer.run(initial_values)
    assert last_evaluation + 1 < 2 * max_functions
    assert exit_code == ExitCode.OPTIMIZER_FINISHED


def test_failed_realizations(config: Any, evaluator: Any, external: str) -> None:
    def _observer(results: tuple[Results, ...]) -> None:
        assert isinstance(results[0], FunctionResults)
        assert results[0].functions is None

    config["backend"]["method"] = f"{external}{_SLSQP}"

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(config, evaluator(functions))
    optimizer.set_results_callback(_observer)
    exit_code = optimizer.run(initial_values)
    assert exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_failed_realizations_constraints(
    config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    def _observer(results: tuple[Results, ...]) -> None:
        assert isinstance(results[0], FunctionResults)
        assert results[0].functions is None

    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    functions = (*test_functions, lambda _0, _1: np.array(np.nan))

    optimizer = BasicOptimizer(config, evaluator(functions))
    optimizer.set_results_callback(_observer)
    exit_code = optimizer.run(initial_values)
    assert exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_all_failed_realizations_not_supported(
    config: Any, evaluator: Any, external: str
) -> None:
    config["realizations"] = {"realization_min_success": 0}

    config["backend"]["method"] = f"{external}{_SLSQP}"

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(config, evaluator(functions))
    exit_code = optimizer.run(initial_values)
    assert exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_user_abort(config: Any, evaluator: Any, external: str) -> None:
    last_evaluation = 0

    def _abort() -> bool:
        nonlocal last_evaluation

        if last_evaluation == 1:
            return True
        last_evaluation += 1
        return False

    config["backend"]["method"] = f"{external}{_SLSQP}"

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_abort_callback(_abort)
    exit_code = optimizer.run(initial_values)
    assert optimizer.results is not None
    assert last_evaluation == 1
    assert exit_code == ExitCode.USER_ABORT


def test_single_perturbation(config: Any, evaluator: Any, external: str) -> None:
    config["gradient"] = {
        "number_of_perturbations": 1,
        "merge_realizations": True,
    }

    config["realizations"] = {"weights": 5 * [1]}
    config["backend"]["method"] = f"{external}{_SLSQP}"

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_external_error(config: Any, evaluator: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["backend"]["options"] = {"ftol": "foo"}
    err = "Input should be a valid number, unable to parse string as a number"
    with pytest.raises(ValueError, match=err):
        BasicOptimizer(config, evaluator())


@pytest.mark.parametrize("use_plugin", [False, True])
def test_objective_with_scaler(
    config: Any,
    evaluator: Any,
    test_functions: Any,
    external: str,
    use_plugin: Any,
) -> None:
    optimizer1 = BasicOptimizer(config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.functions is not None
    variables1 = optimizer1.results.evaluations.variables
    objectives1 = optimizer1.results.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    def function1(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[0](variables, None))

    def function2(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[1](variables, None))

    init1 = test_functions[1](initial_values, None)

    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["objectives"]["transforms"] = [0]
    config["objective_transforms"] = [
        {"method": "scaler", "options": {"scales": [init1, init1]}}
        if use_plugin
        else DefaultObjectiveTransform(
            ObjectiveTransformConfig.model_validate(
                {"method": "scaler", "options": {"scales": [init1, init1]}}
            )
        )
    ]

    checked = False

    def check_value(event: EnOptEvent) -> None:
        nonlocal checked
        results = event.results
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(event.context)
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer2 = BasicOptimizer(config, evaluator([function1, function2]))
    optimizer2._observers.append(  # noqa: SLF001
        (EnOptEventType.FINISHED_EVALUATION, check_value)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(optimizer2.results.evaluations.variables, variables1, atol=0.02)
    assert optimizer2.results.functions is not None
    assert np.allclose(objectives1, optimizer2.results.functions.objectives, atol=0.025)


@pytest.mark.parametrize("use_plugin", [False, True])
def test_objective_with_lazy_scaler(
    config: Any,
    evaluator: Any,
    test_functions: Any,
    external: str,
    use_plugin: Any,
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    optimizer1 = BasicOptimizer(config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.functions is not None
    variables1 = optimizer1.results.evaluations.variables
    objectives1 = optimizer1.results.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["objectives"]["transforms"] = [0]
    config["objective_transforms"] = [
        {"method": "scaler"}
        if use_plugin
        else DefaultObjectiveTransform(
            ObjectiveTransformConfig.model_validate({"method": "scaler"})
        )
    ]

    init1 = test_functions[1](initial_values, None)

    def function1(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[0](variables, None))

    def function2(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[1](variables, None))

    checked = False

    def set_scales(event: EnOptEvent) -> None:
        transform = event.context.objective_transforms[0]
        transform.update([init1, init1])

    def check_value(event: EnOptEvent) -> None:
        nonlocal checked
        results = event.results
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(event.context)
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer2 = BasicOptimizer(config, evaluator([function1, function2]))
    optimizer2._observers.append(  # noqa: SLF001
        (EnOptEventType.FINISHED_EVALUATION, check_value)
    )
    optimizer2._observers.append(  # noqa: SLF001
        (EnOptEventType.START_EVALUATION, set_scales)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(optimizer2.results.evaluations.variables, variables1, atol=0.02)
    assert optimizer2.results.functions is not None
    assert np.allclose(objectives1, optimizer2.results.functions.objectives, atol=0.025)


@pytest.mark.parametrize("use_plugin", [False, True])
def test_nonlinear_constraint_with_scaler(
    config: Any,
    evaluator: Any,
    test_functions: Any,
    external: str,
    use_plugin: Any,
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )
    scales = np.array(functions[-1](initial_values, None), ndmin=1)

    config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    optimizer1 = BasicOptimizer(config, evaluator(functions))
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    config["nonlinear_constraints"]["transforms"] = [0]
    config["nonlinear_constraint_transforms"] = [
        {"method": "scaler", "options": {"scales": scales}}
        if use_plugin
        else DefaultNonlinearConstraintTransform(
            NonlinearConstraintTransformConfig.model_validate(
                {"method": "scaler", "options": {"scales": scales}}
            )
        )
    ]

    context = EnOptContext.model_validate(config)
    assert context.nonlinear_constraints is not None
    assert context.nonlinear_constraints.upper_bounds == 0.4
    bounds = context.nonlinear_constraint_transforms[0].bounds_to_optimizer(
        context.nonlinear_constraints.lower_bounds,
        context.nonlinear_constraints.upper_bounds,
    )
    assert bounds[1] == 0.4 / scales

    check = True

    def check_constraints(event: EnOptEvent) -> None:
        nonlocal check
        results = event.results
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(event.context)
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer2 = BasicOptimizer(config, evaluator(functions))
    optimizer2._observers.append(  # noqa: SLF001
        (EnOptEventType.FINISHED_EVALUATION, check_constraints)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables,
        optimizer1.results.evaluations.variables,
        atol=0.02,
    )
    assert optimizer1.results.functions is not None
    assert optimizer2.results.functions is not None
    assert np.allclose(
        optimizer1.results.functions.objectives,
        optimizer2.results.functions.objectives,
        atol=0.025,
    )


@pytest.mark.parametrize("use_plugin", [False, True])
def test_nonlinear_constraint_with_lazy_scaler(
    config: Any,
    evaluator: Any,
    test_functions: Any,
    external: str,
    use_plugin: Any,
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )
    scales = np.array(functions[-1](initial_values, None), ndmin=1)

    optimizer1 = BasicOptimizer(config, evaluator(functions))
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    config["nonlinear_constraints"]["transforms"] = [0]
    config["nonlinear_constraint_transforms"] = [
        {"method": "scaler"}
        if use_plugin
        else DefaultNonlinearConstraintTransform(
            NonlinearConstraintTransformConfig.model_validate({"method": "scaler"})
        )
    ]

    context = EnOptContext.model_validate(config)
    assert context.nonlinear_constraints is not None
    assert context.nonlinear_constraints.upper_bounds == 0.4
    bounds = context.nonlinear_constraint_transforms[0].bounds_to_optimizer(
        context.nonlinear_constraints.lower_bounds,
        context.nonlinear_constraints.upper_bounds,
    )
    assert bounds[1] == 0.4

    def constraint_function(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[0] + variables[2])

    functions = (*test_functions, constraint_function)

    check = True

    def set_scales(event: EnOptEvent) -> None:
        transform = event.context.nonlinear_constraint_transforms[0]
        transform.update(scales)

    def check_constraints(event: EnOptEvent) -> None:
        nonlocal check
        results = event.results
        context = event.context
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert context.nonlinear_constraints is not None
                _, upper_bounds = context.nonlinear_constraint_transforms[
                    0
                ].bounds_to_optimizer(
                    context.nonlinear_constraints.lower_bounds,
                    context.nonlinear_constraints.upper_bounds,
                )
                assert np.allclose(upper_bounds, 0.4 / scales)
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(event.context)
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer2 = BasicOptimizer(config, evaluator(functions))
    optimizer2._observers.append(  # noqa: SLF001
        (EnOptEventType.FINISHED_EVALUATION, check_constraints)
    )
    optimizer2._observers.append(  # noqa: SLF001
        (EnOptEventType.START_EVALUATION, set_scales)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables,
        optimizer1.results.evaluations.variables,
        atol=0.02,
    )
    assert optimizer1.results.functions is not None
    assert optimizer2.results.functions is not None
    assert np.allclose(
        optimizer1.results.functions.objectives,
        optimizer2.results.functions.objectives,
        atol=0.025,
    )


@pytest.mark.parametrize("use_plugin", [False, True])
@pytest.mark.parametrize("offsets", [None, np.array([1.0, 1.1, 1.2])])
@pytest.mark.parametrize("scales", [None, np.array([2.0, 2.1, 2.2])])
def test_variables_scale_with_scaler(  # noqa: PLR0917
    config: Any,
    evaluator: Any,
    use_plugin: Any,
    offsets: NDArray[np.float64] | None,
    scales: NDArray[np.float64] | None,
    external: str,
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    lower_bounds = np.array([-2.0, -np.inf, -3.0])
    upper_bounds = np.array([np.inf, 1.0, 4.0])

    config["backend"]["max_iterations"] = 20
    config["variables"]["lower_bounds"] = lower_bounds
    config["variables"]["upper_bounds"] = upper_bounds
    config["variables"]["transforms"] = [0]
    config["variable_transforms"] = [
        {"method": "scaler", "options": {"scales": scales, "offsets": offsets}}
        if use_plugin
        else DefaultVariableTransform(
            VariableTransformConfig.model_validate(
                {"method": "scaler", "options": {"scales": scales, "offsets": offsets}}
            )
        )
    ]

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None

    context = EnOptContext.model_validate(config)
    if offsets is not None:
        lower_bounds -= offsets
        upper_bounds -= offsets
    if scales is not None:
        lower_bounds /= scales
        upper_bounds /= scales
    assert np.allclose(context.variables.lower_bounds, lower_bounds)
    assert np.allclose(context.variables.upper_bounds, upper_bounds)
    result = optimizer.results.evaluations.variables
    assert np.allclose(result, [0.0, 0.0, 0.5], atol=0.05)


@pytest.mark.parametrize("use_plugin", [False, True])
def test_variables_scale_linear_constraints_with_scaler(
    config: Any, evaluator: Any, external: str, use_plugin: Any
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    coefficients = [[1, 0, 1], [0, 1, 1]]
    lower_bounds = [1.0, 0.75]
    upper_bounds = [1.0, 0.75]

    config["linear_constraints"] = {
        "coefficients": coefficients,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }

    offsets = np.array([1.0, 1.1, 1.2])
    scales = np.array([2.0, 2.1, 2.2])
    config["variables"]["transforms"] = [0]
    config["variable_transforms"] = [
        {"method": "scaler", "options": {"scales": scales, "offsets": offsets}}
        if use_plugin
        else DefaultVariableTransform(
            VariableTransformConfig.model_validate(
                {"method": "scaler", "options": {"scales": scales, "offsets": offsets}}
            )
        )
    ]

    context = EnOptContext.model_validate(config)
    assert context.linear_constraints is not None
    transformed_coefficients = coefficients * scales
    transformed_scales = np.max(np.abs(transformed_coefficients), axis=-1)
    assert np.allclose(
        context.linear_constraints.coefficients,
        transformed_coefficients / transformed_scales[:, np.newaxis],
    )
    offsets = np.matmul(coefficients, offsets)
    assert np.allclose(
        context.linear_constraints.lower_bounds,
        (lower_bounds - offsets) / transformed_scales,
    )
    assert np.allclose(
        context.linear_constraints.upper_bounds,
        (upper_bounds - offsets) / transformed_scales,
    )

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


def test_check_linear_constraints(config: Any, evaluator: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["linear_constraints"] = {
        "coefficients": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
        "lower_bounds": [0.0, -np.inf, -1.0],
        "upper_bounds": [0.0, 1.0, np.inf],
    }
    optimizer1 = BasicOptimizer(config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None

    config["linear_constraints"]["lower_bounds"] = [0.0, -np.inf, -1.0]
    config["linear_constraints"]["upper_bounds"] = [0.0, 1.0, np.inf]
    optimizer2 = BasicOptimizer(config, evaluator())
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables,
        optimizer2.results.evaluations.variables,
    )

    config["linear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    config["linear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    optimizer3 = BasicOptimizer(config, evaluator())
    optimizer3.run(initial_values)
    assert optimizer3.results is None


def test_check_nonlinear_constraints(
    config: Any, evaluator: Any, test_functions: Any, external: str
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.0, -np.inf, 0.0],
        "upper_bounds": [0.0, 0.0, np.inf],
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0],
        lambda variables, _: variables[0],
        lambda variables, _: variables[0],
    )

    optimizer1 = BasicOptimizer(config, evaluator(test_functions))
    optimizer1.run(initial_values)
    assert optimizer1.results is not None

    # Flipping the bounds should still work:
    config["nonlinear_constraints"]["lower_bounds"] = [0.0, -np.inf, 0.0]
    config["nonlinear_constraints"]["upper_bounds"] = [0.0, 0.0, np.inf]
    optimizer2 = BasicOptimizer(config, evaluator(test_functions))
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables,
        optimizer2.results.evaluations.variables,
    )

    config["nonlinear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    config["nonlinear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    optimizer3 = BasicOptimizer(config, evaluator(test_functions))
    optimizer3.run(initial_values)
    assert optimizer3.results is None


def test_optimizer_variables_subset(config: Any, evaluator: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    config["variables"]["mask"] = [True, False, True]

    def assert_gradient(results: tuple[Results, ...]) -> None:
        for item in results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.target_objective[1] == 0.0
                assert np.all(item.gradients.objectives[:, 1] == 0.0)

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_results_callback(assert_gradient)
    optimizer.run([0.0, 1.0, 0.1])
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 1.0, 0.5], atol=0.02
    )


def test_optimizer_variables_subset_linear_constraints(
    config: Any, evaluator: Any, external: str
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem. The
    # second and third constraints are dropped because they involve variables
    # that are not optimized.
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "lower_bounds": [1.0, 1.0, 2.0],
        "upper_bounds": [1.0, 1.0, 2.0],
    }
    config["variables"]["mask"] = [True, False, True]

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run([0.0, 1.0, 0.1])
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 1.0, 0.75], atol=0.02
    )


def test_parallelize(config: Any, evaluator: Any, external: str) -> None:
    config["optimizer"] = {}
    config["backend"] = {
        "method": f"{external}{_DIFFERENTIAL_EVOLUTION}",
        "max_iterations": 15,
        "options": {"rng": 123, "tol": 1e-10},
    }
    config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]

    config["backend"]["parallel"] = False
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run([0.2, *initial_values[1:]])
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.15, 0.0, 0.2], atol=3e-2
    )

    config["backend"]["parallel"] = True
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run([0.2, *initial_values[1:]])
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.15, 0.0, 0.2], atol=3e-2
    )


def test_rng(config: Any, evaluator: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    optimizer1 = BasicOptimizer(config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )

    optimizer2 = BasicOptimizer(config, evaluator())
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert np.all(
        optimizer2.results.evaluations.variables
        == optimizer2.results.evaluations.variables
    )

    config["variables"]["seed"] = (1, DEFAULT_SEED)
    optimizer3 = BasicOptimizer(config, evaluator())
    optimizer3.run(initial_values)
    assert optimizer3.results is not None
    assert np.allclose(
        optimizer3.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert not np.all(
        optimizer3.results.evaluations.variables
        == optimizer1.results.evaluations.variables
    )


def test_arbitrary_objective_weights(
    config: Any, evaluator: Any, external: str, test_functions: Any
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    new_functions = (
        *test_functions,
        lambda variables, _: test_functions[1](variables, None),
    )

    config["objectives"]["weights"] = [0.75, 0.25, -0.25]
    optimizer = BasicOptimizer(config, evaluator(new_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert not np.allclose(
        optimizer.results.evaluations.variables, [0, 0, 0.5], atol=0.02
    )

    config["objectives"]["weights"] = [0.75, 0.25, 0.0]
    optimizer = BasicOptimizer(config, evaluator(new_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(optimizer.results.evaluations.variables, [0, 0, 0.5], atol=0.02)

    config["objectives"]["weights"] = [0.75, 0.5, -0.25]
    optimizer = BasicOptimizer(config, evaluator(new_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(optimizer.results.evaluations.variables, [0, 0, 0.5], atol=0.02)

    config["objectives"]["weights"] = [-0.75, -0.25]
    with pytest.raises(ValidationError, match="The sum of weights is not positive"):
        EnOptContext.model_validate(config)
