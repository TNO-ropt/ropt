# ruff: file-ignore[float-equality-comparison]

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.components.event_handlers import CallbackHandler
from ropt.config import LinearConstraintsConfig
from ropt.config.constants import DEFAULT_SEED
from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, ExitCode
from ropt.results import FunctionResults, GradientResults
from ropt.simple import EvaluateResult, optimize
from ropt.utils import validate_backend_options

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

initial_values = np.array([0.0, 0.0, 0.1])


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
            "variable_count": initial_values.size,
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_basic_run(config: Any, eval_func: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0, 0, 0.5], atol=0.02)


def test_invalid_options(config: Any, external: str) -> None:
    config["backend"]["options"] = {"ftol": 0.1, "foo": 1}
    config["backend"]["method"] = f"{external}{_SLSQP}"

    method = config["backend"]["method"]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`"
    ):
        validate_backend_options(method, config["backend"]["options"])


def test_common_options(config: Any, external: str) -> None:
    config["backend"]["options"] = {"disp": True}
    config["backend"]["method"] = f"{external}{_SLSQP}"
    validate_backend_options(config["backend"]["method"], config["backend"]["options"])


def test_max_functions_exceeded(config: Any, eval_func: Any, external: str) -> None:
    last_evaluation = 0

    def track_results(_: EnOptEvent) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 2
    config["optimizer"]["max_functions"] = max_functions
    config["backend"]["method"] = f"{external}{_SLSQP}"
    result = optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION}, callback=track_results
            )
        ],
    )
    assert last_evaluation == max_functions + 1
    assert result.exit_code == ExitCode.MAX_FUNCTIONS_REACHED


def test_max_batches_exceeded(config: Any, eval_func: Any, external: str) -> None:
    last_evaluation = 0

    def track_results(_: EnOptEvent) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_batches = 2
    config["optimizer"]["max_batches"] = max_batches
    config["backend"]["method"] = f"{external}{_SLSQP}"
    result = optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION}, callback=track_results
            )
        ],
    )
    assert last_evaluation == max_batches
    assert result.exit_code == ExitCode.MAX_BATCHES_REACHED


def test_max_functions_not_exceeded(config: Any, eval_func: Any, external: str) -> None:
    last_evaluation = 0

    def track_results(_: EnOptEvent) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 100
    config["optimizer"]["max_functions"] = max_functions
    config["gradient"] = {"evaluation_policy": "separate"}
    config["backend"]["method"] = f"{external}{_SLSQP}"
    result = optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION}, callback=track_results
            )
        ],
    )
    assert last_evaluation + 1 < 2 * max_functions
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


def test_failed_realizations(config: Any, eval_func: Any, external: str) -> None:
    def _observer(item: EvaluateResult) -> None:
        assert item.target_objective is None

    config["backend"]["method"] = f"{external}{_SLSQP}"

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    result = optimize(config, initial_values, eval_func(functions), report=_observer)
    assert result.exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_failed_realizations_constraints(
    config: Any, eval_func: Any, test_functions: Any, external: str
) -> None:
    def _observer(item: EvaluateResult) -> None:
        assert item.target_objective is None

    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    result = optimize(
        config,
        initial_values,
        eval_func(test_functions, [lambda _0, _1: np.nan]),
        report=_observer,
    )
    assert result.exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_single_perturbation(config: Any, eval_func: Any, external: str) -> None:
    config["gradient"] = {
        "number_of_perturbations": 1,
        "merge_realizations": True,
    }

    config["realizations"] = {"weights": 5 * [1]}
    config["backend"]["method"] = f"{external}{_SLSQP}"

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_external_error(config: Any, eval_func: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["backend"]["options"] = {"ftol": "foo"}
    err = "Input should be a valid number, unable to parse string as a number"
    with pytest.raises(ValueError, match=err):
        optimize(config, initial_values, eval_func())


def test_objective_with_scales(
    config: Any,
    eval_func: Any,
    test_functions: Any,
    external: str,
) -> None:
    result1 = optimize(config, initial_values, eval_func())
    assert result1.variables is not None
    assert result1.objectives is not None
    variables1 = result1.variables
    objectives1 = result1.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    def function1(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[0](variables, None))

    def function2(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[1](variables, None))

    init1 = test_functions[1](initial_values, None)

    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["objectives"]["scales"] = [init1, init1]

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

    result2 = optimize(
        config,
        initial_values,
        eval_func([function1, function2]),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION}, callback=check_value
            )
        ],
    )
    assert checked
    assert result2.variables is not None
    assert np.allclose(result2.variables, variables1, atol=0.02)
    assert result2.objectives is not None
    assert np.allclose(objectives1, result2.objectives, atol=0.025)


def test_objective_with_auto_scale(
    config: Any,
    eval_func: Any,
    test_functions: Any,
    external: str,
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    result1 = optimize(config, initial_values, eval_func())
    assert result1.variables is not None
    assert result1.objectives is not None
    variables1 = result1.variables
    objectives1 = result1.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    config["objectives"]["auto_scale"] = True

    def function1(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[0](variables, None))

    def function2(variables: NDArray[np.float64], _: Any) -> float:
        return float(test_functions[1](variables, None))

    # A single scale, the weighted total of the objectives at the initial point.
    weights = np.array(config["objectives"]["weights"])
    weights /= weights.sum()
    initial = np.array([f(initial_values, None) for f in test_functions])
    scale = abs(np.dot(initial, weights))

    checked = False

    def check_value(event: EnOptEvent) -> None:
        nonlocal checked
        results = event.results
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert np.allclose(event.context.get_objective_scales(), scale)
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives, initial / scale)
                assert np.allclose(item.functions.target_objective, 1.0)
                transformed = item.transform_from_optimizer(event.context)
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives, initial)

    result2 = optimize(
        config,
        initial_values,
        eval_func([function1, function2]),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION}, callback=check_value
            ),
        ],
    )
    assert checked
    assert result2.variables is not None
    assert np.allclose(result2.variables, variables1, atol=0.02)
    assert result2.objectives is not None
    # Rescaling the objective changes the steps the optimizer takes, so within
    # the same budget it stops at a slightly different point.
    assert np.allclose(objectives1, result2.objectives, atol=0.05)


def test_nonlinear_constraint_with_scales(
    config: Any,
    eval_func: Any,
    test_functions: Any,
    external: str,
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    def constraint_function(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[0] + variables[2])

    scales = np.array(constraint_function(initial_values, None), ndmin=1)

    config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    result1 = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function]),
    )
    assert result1.variables is not None
    assert result1.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert result1.variables[[0, 2]].sum() < 0.4 + 1e-5

    config["nonlinear_constraints"]["scales"] = scales

    context = EnOptContext.model_validate(config)
    assert context.nonlinear_constraints is not None
    assert context.nonlinear_constraints.upper_bounds == 0.4
    bounds = context.get_nonlinear_constraint_bounds()
    assert bounds is not None
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

    result2 = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function]),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION},
                callback=check_constraints,
            )
        ],
    )
    assert not check
    assert result2.variables is not None
    assert np.allclose(result2.variables, result1.variables, atol=0.02)
    assert result1.objectives is not None
    assert result2.objectives is not None
    assert np.allclose(result1.objectives, result2.objectives, atol=0.025)


def test_nonlinear_constraint_with_auto_scale(
    config: Any,
    eval_func: Any,
    test_functions: Any,
    external: str,
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"

    config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    def constraint_function(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[0] + variables[2])

    scales = np.array(constraint_function(initial_values, None), ndmin=1)

    result1 = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function]),
    )
    assert result1.variables is not None
    assert result1.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert result1.variables[[0, 2]].sum() < 0.4 + 1e-5

    config["nonlinear_constraints"]["auto_scale"] = True

    context = EnOptContext.model_validate(config)
    assert context.nonlinear_constraints is not None
    assert context.nonlinear_constraints.upper_bounds == 0.4
    # Before the first batch the scales are still one.
    bounds = context.get_nonlinear_constraint_bounds()
    assert bounds is not None
    assert bounds[1] == 0.4

    check = True

    def check_constraints(event: EnOptEvent) -> None:
        nonlocal check
        results = event.results
        context = event.context
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                constraint_scales = context.get_constraint_scales()
                assert constraint_scales is not None
                assert np.allclose(constraint_scales, scales)
                bounds = context.get_nonlinear_constraint_bounds()
                assert bounds is not None
                assert np.allclose(bounds[1], 0.4 / scales)
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(event.context)
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    result2 = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function]),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION},
                callback=check_constraints,
            ),
        ],
    )
    assert not check
    assert result2.variables is not None
    assert np.allclose(result2.variables, result1.variables, atol=0.02)
    assert result1.objectives is not None
    assert result2.objectives is not None
    assert np.allclose(result1.objectives, result2.objectives, atol=0.025)


@pytest.mark.parametrize("offsets", [None, np.array([1.0, 1.1, 1.2])])
@pytest.mark.parametrize("scales", [None, np.array([2.0, 2.1, 2.2])])
def test_variables_are_scaled_and_offset(
    config: Any,
    eval_func: Any,
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
    if scales is not None:
        config["variables"]["scales"] = scales
    if offsets is not None:
        config["variables"]["offsets"] = offsets

    opt_result = optimize(config, initial_values, eval_func())
    assert opt_result.variables is not None

    context = EnOptContext.model_validate(config)
    if offsets is not None:
        lower_bounds -= offsets
        upper_bounds -= offsets
    if scales is not None:
        lower_bounds /= scales
        upper_bounds /= scales
    assert np.allclose(context.variables.lower_bounds, lower_bounds)
    assert np.allclose(context.variables.upper_bounds, upper_bounds)
    # The optimum is reported in the user domain, so it is where it always was.
    assert np.allclose(opt_result.variables, [0.0, 0.0, 0.5], atol=0.05)


def test_scaled_variables_change_the_linear_constraints(
    config: Any, eval_func: Any, external: str
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
    config["variables"]["scales"] = scales
    config["variables"]["offsets"] = offsets
    config["linear_constraints"]["auto_scale"] = True

    context = EnOptContext.model_validate(config)
    assert isinstance(context.linear_constraints, LinearConstraintsConfig)
    transformed_coefficients = coefficients * scales
    shift = np.matmul(coefficients, offsets)
    transformed_scales = np.maximum(
        np.max(np.abs(transformed_coefficients), axis=-1),
        np.maximum(np.abs(lower_bounds - shift), np.abs(upper_bounds - shift)),
    )
    assert np.allclose(
        context.linear_constraints.coefficients,
        transformed_coefficients / transformed_scales[:, np.newaxis],
    )
    assert np.allclose(
        context.linear_constraints.lower_bounds,
        (lower_bounds - shift) / transformed_scales,
    )
    assert np.allclose(
        context.linear_constraints.upper_bounds,
        (upper_bounds - shift) / transformed_scales,
    )

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 0.0, 0.75], atol=0.02)


def test_check_linear_constraints(config: Any, eval_func: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["linear_constraints"] = {
        "coefficients": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
        "lower_bounds": [0.0, -np.inf, -1.0],
        "upper_bounds": [0.0, 1.0, np.inf],
    }
    result1 = optimize(config, initial_values, eval_func())
    assert result1.variables is not None

    config["linear_constraints"]["lower_bounds"] = [0.0, -np.inf, -1.0]
    config["linear_constraints"]["upper_bounds"] = [0.0, 1.0, np.inf]
    result2 = optimize(config, initial_values, eval_func())
    assert result2.variables is not None
    assert np.allclose(result1.variables, result2.variables)

    config["linear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    config["linear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    result3 = optimize(config, initial_values, eval_func())
    assert result3.variables is None


def test_check_nonlinear_constraints(
    config: Any, eval_func: Any, test_functions: Any, external: str
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.0, -np.inf, 0.0],
        "upper_bounds": [0.0, 0.0, np.inf],
    }

    constraint_functions = (
        lambda variables, _: variables[0],
        lambda variables, _: variables[0],
        lambda variables, _: variables[0],
    )

    result1 = optimize(
        config, initial_values, eval_func(test_functions, constraint_functions)
    )
    assert result1.variables is not None

    # Flipping the bounds should still work:
    config["nonlinear_constraints"]["lower_bounds"] = [0.0, -np.inf, 0.0]
    config["nonlinear_constraints"]["upper_bounds"] = [0.0, 0.0, np.inf]
    result2 = optimize(config, initial_values, eval_func(test_functions))
    assert result2.variables is not None
    assert np.allclose(result1.variables, result2.variables)

    config["nonlinear_constraints"]["lower_bounds"] = [1.0, -np.inf, 1.0]
    config["nonlinear_constraints"]["upper_bounds"] = [1.0, -1.0, np.inf]

    result3 = optimize(
        config, initial_values, eval_func(test_functions, constraint_functions)
    )
    assert result3.variables is None


def test_optimizer_variables_subset(config: Any, eval_func: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    config["variables"]["mask"] = [True, False, True]

    def assert_gradient(event: EnOptEvent) -> None:
        for item in event.results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.target_objective[1] == 0.0
                assert np.all(item.gradients.objectives[:, 1] == 0.0)

    result = optimize(
        config,
        [0.0, 1.0, 0.1],
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION},
                callback=assert_gradient,
            )
        ],
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 1.0, 0.5], atol=0.02)


def test_optimizer_variables_subset_linear_constraints(
    config: Any, eval_func: Any, external: str
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    # The second constraint only involves the fixed variable.
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0]],
        "lower_bounds": [1.0, 1.0],
        "upper_bounds": [1.0, 1.0],
    }
    config["variables"]["mask"] = [True, False, True]

    result = optimize(config, [0.0, 1.0, 0.1], eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 1.0, 0.75], atol=0.02)


def test_optimizer_variables_subset_linear_constraints_offset(
    config: Any, eval_func: Any, external: str
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 1, 0]],
        "lower_bounds": [1.0, -np.inf],
        "upper_bounds": [1.0, 1.15],
    }
    config["variables"]["mask"] = [True, False, True]

    result = optimize(config, [0.0, 1.0, 0.1], eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.15, 1.0, 0.85], atol=0.02)


def test_parallelize(config: Any, eval_func: Any, external: str) -> None:
    config["optimizer"] = {}
    config["backend"] = {
        "method": f"{external}{_DIFFERENTIAL_EVOLUTION}",
        "max_iterations": 15,
        "options": {"rng": 123, "tol": 1e-10},
    }
    config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]

    config["backend"]["parallel"] = False
    result = optimize(config, [0.2, *initial_values[1:]], eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.15, 0.0, 0.2], atol=3e-2)

    config["backend"]["parallel"] = True
    result = optimize(config, [0.2, *initial_values[1:]], eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.15, 0.0, 0.2], atol=3e-2)


def test_rng(config: Any, eval_func: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    result1 = optimize(config, initial_values, eval_func())
    assert result1.variables is not None
    assert np.allclose(result1.variables, [0.0, 0.0, 0.5], atol=0.02)

    result2 = optimize(config, initial_values, eval_func())
    assert result2.variables is not None
    assert np.allclose(result2.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert np.all(result2.variables == result2.variables)

    config["variables"]["seed"] = (1, DEFAULT_SEED)
    result3 = optimize(config, initial_values, eval_func())
    assert result3.variables is not None
    assert np.allclose(result3.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.all(result3.variables == result1.variables)


def test_zero_objective_weight_disables_an_objective(
    config: Any, eval_func: Any, external: str, test_functions: Any
) -> None:
    config["backend"]["method"] = f"{external}{_SLSQP}"
    new_functions = (
        *test_functions,
        lambda variables, _: test_functions[1](variables, None),
    )

    config["objectives"]["weights"] = [0.75, 0.25, 0.0]
    result = optimize(config, initial_values, eval_func(new_functions))
    assert result.variables is not None
    assert np.allclose(result.variables, [0, 0, 0.5], atol=0.02)
