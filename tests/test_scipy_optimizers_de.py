# ruff: file-ignore[float-equality-comparison]

from typing import Any

import numpy as np
import pytest

from ropt.exceptions import UnsupportedError
from ropt.simple import optimize
from ropt.utils import validate_backend_options

pytestmark = [pytest.mark.slow]

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "backend": {
            "method": "differential_evolution",
            "convergence_tolerance": 1e-5,
            "options": {"rng": 1},
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_scipy_required_constraints_bounds_de(config: Any, eval_func: Any) -> None:
    with pytest.raises(UnsupportedError, match="requires bound constraints"):
        optimize(config, initial_values, eval_func())


def test_scipy_bound_constraints_de(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]

    validate_backend_options("differential_evolution", config["backend"]["options"])

    result = optimize(config, [0.2, *initial_values[1:]], eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.15, 0.0, 0.2], atol=0.03)


def test_scipy_bound_constraints_differential_evolution_de(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]

    config["realizations"] = {"realization_min_success": 0}
    result1 = optimize(config, [0.2, *initial_values[1:]], eval_func())
    assert result1.variables is not None
    assert np.allclose(result1.variables, [0.15, 0.0, 0.2], atol=0.03)

    counter = 0

    def _add_nan(x: Any, c: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x, c)

    result2 = optimize(
        config, [0.2, *initial_values[1:]], eval_func((_add_nan, test_functions[1]))
    )
    assert result2.variables is not None
    assert np.allclose(result2.variables, [0.15, 0.0, 0.2], atol=0.03)
    assert not np.all(result1.variables == result2.variables)


def test_scipy_eq_linear_constraints_de(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    # The result should be [0.25, 0.0, 0.75], but DE appears to have
    # difficulties with linear equality equations. Therefore, we just test if it
    # does not violate them.
    assert result.variables[0] + result.variables[2] == pytest.approx(1.0, abs=0.02)
    assert result.variables[1] + result.variables[2] == pytest.approx(0.75, abs=0.02)


def test_scipy_ge_linear_constraints_de(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_le_linear_constraints_de(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_le_ge_linear_constraints_de(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_le_ge_linear_constraints_two_sided_de(
    config: Any, eval_func: Any
) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.03)

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.03)


def test_scipy_eq_nonlinear_constraints_de(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    def constraint_function(variables: Any, _: Any) -> Any:
        return variables[0] + variables[2]

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 0.0, 0.75], atol=0.03)


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_scipy_ineq_nonlinear_constraints_de(
    config: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }

    weight = 1.0 if upper_bounds == 0.4 else -1.0

    def constraint_function(variables: Any, _: Any) -> Any:
        return weight * variables[0] + weight * variables[2]

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_ineq_nonlinear_constraints_two_sided_de(
    config: Any,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    def constraint_function(variables: Any, _: Any) -> Any:
        return variables[0] + variables[2]

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.03)
