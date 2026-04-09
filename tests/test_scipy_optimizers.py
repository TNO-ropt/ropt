# ruff: noqa: RUF069

from typing import Any, Literal

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.backend.scipy import (
    _CONSTRAINT_REQUIRES_BOUNDS,
    _CONSTRAINT_SUPPORT_BOUNDS,
    _CONSTRAINT_SUPPORT_LINEAR_EQ,
    _CONSTRAINT_SUPPORT_LINEAR_INEQ,
    _CONSTRAINT_SUPPORT_NONLINEAR_EQ,
    _CONSTRAINT_SUPPORT_NONLINEAR_INEQ,
    SUPPORTED_SCIPY_METHODS,
)
from ropt.results import Results
from ropt.workflow import BasicOptimizer, validate_backend_options

_REQUIRES_BOUNDS = _CONSTRAINT_REQUIRES_BOUNDS - {"differential_evolution"}
_SUPPORTS_BOUNDS = _CONSTRAINT_SUPPORT_BOUNDS - {"differential_evolution"}
_SUPPORTS_LINEAR_EQ = _CONSTRAINT_SUPPORT_LINEAR_EQ - {"differential_evolution"}
_SUPPORTS_LINEAR_INEQ = _CONSTRAINT_SUPPORT_LINEAR_INEQ - {"differential_evolution"}
_SUPPORTS_NONLINEAR_EQ = _CONSTRAINT_SUPPORT_NONLINEAR_EQ - {"differential_evolution"}
_SUPPORTS_NONLINEAR_INEQ = _CONSTRAINT_SUPPORT_NONLINEAR_INEQ - {
    "differential_evolution"
}
_SUPPORTED = SUPPORTED_SCIPY_METHODS - {"differential_evolution"}

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.0001,
        },
        "backend": {
            "convergence_tolerance": 1e-5,
            "max_iterations": 25,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_scipy_invalid_options(config: Any) -> None:
    config["backend"]["options"] = {"foo": 1}
    config["backend"]["method"] = "slsqp"

    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`"
    ):
        validate_backend_options("slsqp", config["backend"]["options"])


def test_scipy_invalid_options_type(config: Any) -> None:
    config["backend"]["options"] = ["foo=1"]
    config["backend"]["method"] = "slsqp"

    with pytest.raises(ValueError, match="SciPy backend options must be a dictionary"):
        validate_backend_options("slsqp", config["backend"]["options"])


@pytest.mark.parametrize("method", sorted(_SUPPORTED - _REQUIRES_BOUNDS))
def test_scipy_unconstrained(config: Any, method: str, evaluator: Any) -> None:
    config["backend"]["method"] = method

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    # Some methods are supported, but not reliable in this test.
    if method != "newton-cg":
        assert np.allclose(
            optimizer.results.evaluations.variables, [0, 0, 0.5], atol=0.02
        )


@pytest.mark.parametrize("method", sorted(_SUPPORTED - _SUPPORTS_BOUNDS))
def test_scipy_unsupported_constraints_bounds(
    config: Any, method: str, evaluator: Any
) -> None:
    config["backend"]["method"] = method
    config["variables"]["lower_bounds"] = [0.1, -np.inf, -np.inf]
    with pytest.raises(NotImplementedError, match="does not support bound constraints"):
        BasicOptimizer(config, evaluator()).run(initial_values)


@pytest.mark.parametrize("method", sorted(_REQUIRES_BOUNDS))
def test_scipy_required_constraints_bounds(
    config: Any, method: str, evaluator: Any
) -> None:
    config["backend"]["method"] = method
    with pytest.raises(NotImplementedError, match="requires bound constraints"):
        BasicOptimizer(config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_LINEAR_EQ),
)
def test_scipy_unsupported_constraints_linear_eq(
    config: Any, method: str, evaluator: Any
) -> None:
    config["backend"]["method"] = method
    config["linear_constraints"] = {
        "coefficients": [[1.0, 0.0, 1.0]],
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }
    with pytest.raises(
        NotImplementedError, match="does not support linear equality constraints"
    ):
        BasicOptimizer(config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_LINEAR_INEQ),
)
@pytest.mark.parametrize(("lower_bounds", "upper_bounds"), [(-np.inf, 1), (1, np.inf)])
def test_scipy_unsupported_constraints_linear_ineq(
    config: Any,
    method: str,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
) -> None:
    config["backend"]["method"] = method
    config["linear_constraints"] = {
        "coefficients": [[1.0, 0.0, 1.0]],
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    with pytest.raises(
        NotImplementedError, match="does not support linear inequality constraints"
    ):
        BasicOptimizer(config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_NONLINEAR_EQ),
)
def test_scipy_unsupported_constraints_nonlinear_eq(
    config: Any, method: str, evaluator: Any
) -> None:
    config["backend"]["method"] = method
    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    with pytest.raises(
        NotImplementedError, match="does not support non-linear equality constraints"
    ):
        BasicOptimizer(config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED - _SUPPORTS_NONLINEAR_INEQ),
)
@pytest.mark.parametrize(("lower_bounds", "upper_bounds"), [(-np.inf, 1), (1, np.inf)])
def test_scipy_unsupported_constraints_nonlinear_ineq(
    config: Any,
    method: str,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
) -> None:
    config["backend"]["method"] = method
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    with pytest.raises(
        NotImplementedError, match="does not support non-linear inequality constraints"
    ):
        BasicOptimizer(config, evaluator()).run(initial_values)


@pytest.mark.parametrize("method", sorted(_SUPPORTS_BOUNDS))
def test_scipy_bound_constraints(config: Any, method: str, evaluator: Any) -> None:
    config["backend"]["method"] = method
    config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run([0.2, *initial_values[1:]])
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_EQ))
def test_scipy_eq_linear_constraints(config: Any, method: str, evaluator: Any) -> None:
    if method in _SUPPORTS_BOUNDS:
        config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_ge_linear_constraints(config: Any, method: str, evaluator: Any) -> None:
    if method in _SUPPORTS_BOUNDS:
        config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_le_linear_constraints(config: Any, method: str, evaluator: Any) -> None:
    if method in _SUPPORTS_BOUNDS:
        config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_le_ge_linear_constraints(
    config: Any, method: str, evaluator: Any
) -> None:
    if method in _SUPPORTS_BOUNDS:
        config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_LINEAR_INEQ))
def test_scipy_le_ge_linear_constraints_two_sided(
    config: Any, method: str, evaluator: Any
) -> None:
    if method in _SUPPORTS_BOUNDS:
        config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_NONLINEAR_EQ))
def test_scipy_eq_nonlinear_constraints(
    config: Any,
    method: str,
    evaluator: Any,
    test_functions: Any,
) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_NONLINEAR_INEQ))
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_scipy_ineq_nonlinear_constraints(  # noqa: PLR0917
    config: Any,
    method: str,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    if method in _SUPPORTS_BOUNDS:
        config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: weight * variables[0] + weight * variables[2],
    )

    optimizer = BasicOptimizer(config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


@pytest.mark.parametrize("method", sorted(_SUPPORTS_NONLINEAR_INEQ))
def test_scipy_ineq_nonlinear_constraints_two_sided(
    config: Any,
    method: str,
    evaluator: Any,
    test_functions: Any,
) -> None:
    if method in _SUPPORTS_BOUNDS:
        config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["backend"]["method"] = method
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )


def test_scipy_options(config: Any, evaluator: Any) -> None:
    config["backend"]["method"] = "Nelder-Mead"
    config["backend"]["options"] = {"maxfev": 10}

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert pytest.approx(optimizer.results.evaluations.variables[2], abs=0.025) != 0.5


@pytest.mark.parametrize("method", sorted(_SUPPORTED - _REQUIRES_BOUNDS))
def test_scipy_evaluation_policy_separate(
    config: Any, method: str, evaluator: Any
) -> None:
    config["backend"]["method"] = method
    config["gradient"] = {"evaluation_policy": "separate"}

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    # Some methods are supported, but not reliable in this test.
    if method != "newton-cg":
        assert np.allclose(
            optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
        )


@pytest.mark.parametrize("evaluation_policy", ["speculative", "auto"])
def test_scipy_speculative(
    config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    config["backend"]["method"] = "slsqp"
    config["gradient"] = {"evaluation_policy": evaluation_policy}

    def _observer(results: tuple[Results, ...]) -> None:
        assert len(results) == 2 if evaluation_policy == "speculative" else 1

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_results_callback(_observer)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
