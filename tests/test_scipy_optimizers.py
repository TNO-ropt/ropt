from typing import Any, Literal

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.optimization import BasicOptimizer
from ropt.plugins import PluginManager
from ropt.plugins.optimizer.scipy import (
    _CONSTRAINT_REQUIRES_BOUNDS,
    _CONSTRAINT_SUPPORT_BOUNDS,
    _CONSTRAINT_SUPPORT_LINEAR_EQ,
    _CONSTRAINT_SUPPORT_LINEAR_INEQ,
    _CONSTRAINT_SUPPORT_NONLINEAR_EQ,
    _CONSTRAINT_SUPPORT_NONLINEAR_INEQ,
    _SUPPORTED_METHODS,
)
from ropt.results import Results

_REQUIRES_BOUNDS = _CONSTRAINT_REQUIRES_BOUNDS - {"differential_evolution"}
_SUPPORTS_BOUNDS = _CONSTRAINT_SUPPORT_BOUNDS - {"differential_evolution"}
_SUPPORTS_LINEAR_EQ = _CONSTRAINT_SUPPORT_LINEAR_EQ - {"differential_evolution"}
_SUPPORTS_LINEAR_INEQ = _CONSTRAINT_SUPPORT_LINEAR_INEQ - {"differential_evolution"}
_SUPPORTS_NONLINEAR_EQ = _CONSTRAINT_SUPPORT_NONLINEAR_EQ - {"differential_evolution"}
_SUPPORTS_NONLINEAR_INEQ = _CONSTRAINT_SUPPORT_NONLINEAR_INEQ - {
    "differential_evolution"
}
_SUPPORTED = _SUPPORTED_METHODS - {"differential_evolution"}

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "optimizer": {
            "tolerance": 1e-4,
            "max_iterations": 25,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_scipy_invalid_options(enopt_config: Any) -> None:
    enopt_config["optimizer"]["options"] = {"foo": 1}
    enopt_config["optimizer"]["method"] = "slsqp"

    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`"
    ):
        PluginManager().get_plugin("optimizer", "slsqp").validate_options(
            "slsqp", enopt_config["optimizer"]["options"]
        )


def test_scipy_invalid_options_type(enopt_config: Any) -> None:
    enopt_config["optimizer"]["options"] = ["foo=1"]
    enopt_config["optimizer"]["method"] = "slsqp"

    with pytest.raises(
        ValueError, match="SciPy optimizer options must be a dictionary"
    ):
        PluginManager().get_plugin("optimizer", "slsqp").validate_options(
            "slsqp", enopt_config["optimizer"]["options"]
        )


@pytest.mark.parametrize("method", sorted(_SUPPORTED - _REQUIRES_BOUNDS))
def test_scipy_unconstrained(enopt_config: Any, method: str, evaluator: Any) -> None:
    enopt_config["optimizer"]["method"] = method

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    # Some methods are supported, but not reliable in this test.
    if method != "newton-cg":
        assert np.allclose(variables, [0, 0, 0.5], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTED - _SUPPORTS_BOUNDS))
def test_scipy_unsupported_constraints_bounds(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["variables"]["lower_bounds"] = [0.1, -np.inf, -np.inf]
    with pytest.raises(NotImplementedError, match="does not support bound constraints"):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize("method", sorted(_REQUIRES_BOUNDS))
def test_scipy_required_constraints_bounds(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    with pytest.raises(NotImplementedError, match="requires bound constraints"):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_LINEAR_EQ),
)
def test_scipy_unsupported_constraints_linear_eq(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1.0, 0.0, 1.0]],
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }
    with pytest.raises(
        NotImplementedError, match="does not support linear equality constraints"
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_LINEAR_INEQ),
)
@pytest.mark.parametrize(("lower_bounds", "upper_bounds"), [(-np.inf, 1), (1, np.inf)])
def test_scipy_unsupported_constraints_linear_ineq(
    enopt_config: Any,
    method: str,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1.0, 0.0, 1.0]],
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    with pytest.raises(
        NotImplementedError, match="does not support linear inequality constraints"
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_NONLINEAR_EQ),
)
def test_scipy_unsupported_constraints_nonlinear_eq(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    with pytest.raises(
        NotImplementedError, match="does not support non-linear equality constraints"
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_NONLINEAR_INEQ),
)
@pytest.mark.parametrize(("lower_bounds", "upper_bounds"), [(-np.inf, 1), (1, np.inf)])
def test_scipy_unsupported_constraints_nonlinear_ineq(
    enopt_config: Any,
    method: str,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    with pytest.raises(
        NotImplementedError, match="does not support non-linear inequality constraints"
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_BOUNDS))
def test_scipy_bound_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    variables = (
        BasicOptimizer(enopt_config, evaluator())
        .run([0.2, *initial_values[1:]])
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_EQ))
def test_scipy_eq_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_ge_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_le_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_le_ge_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_le_ge_linear_constraints_two_sided(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_NONLINEAR_EQ))
def test_scipy_eq_nonlinear_constraints(
    enopt_config: Any,
    method: str,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_NONLINEAR_INEQ))
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_scipy_ineq_nonlinear_constraints(
    enopt_config: Any,
    method: str,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: weight * variables[0] + weight * variables[2],
    )

    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_NONLINEAR_INEQ))
def test_scipy_ineq_nonlinear_constraints_two_sided(
    enopt_config: Any,
    method: str,
    evaluator: Any,
    test_functions: Any,
) -> None:
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTS_NONLINEAR_INEQ & _SUPPORTS_NONLINEAR_EQ),
)
def test_scipy_le_ge_nonlinear_constraints(
    enopt_config: Any,
    method: str,
    evaluator: Any,
    test_functions: Any,
) -> None:
    # These constraints together force the first two variables to be zero,
    # while the last one is free to fit the function.
    if method in _SUPPORTS_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["variables"]["lower_bounds"] = 0.0
    enopt_config["optimizer"]["method"] = method

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.4, 0.0],
        "upper_bounds": [np.inf, 0.0],
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] - variables[1],
    )

    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.4], atol=0.02)


def test_scipy_options(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["method"] = "Nelder-Mead"
    enopt_config["optimizer"]["options"] = {"maxfev": 10}

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert pytest.approx(variables[2], abs=0.025) != 0.5


@pytest.mark.parametrize("method", sorted(_SUPPORTED - _REQUIRES_BOUNDS))
def test_scipy_evaluation_policy_separate(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["gradient"] = {"evaluation_policy": "separate"}

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    # Some methods are supported, but not reliable in this test.
    if method != "newton-cg":
        assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize("evaluation_policy", ["speculative", "auto"])
def test_scipy_speculative(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    enopt_config["optimizer"]["method"] = "slsqp"
    enopt_config["gradient"] = {"evaluation_policy": evaluation_policy}

    def _observer(results: tuple[Results, ...]) -> None:
        assert len(results) == 2 if evaluation_policy == "speculative" else 1

    variables = (
        BasicOptimizer(enopt_config, evaluator())
        .set_results_callback(_observer)
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)
