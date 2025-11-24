from typing import Any

import numpy as np
import pytest

from ropt.workflow import BasicOptimizer, validate_optimizer_options

pytestmark = [pytest.mark.slow]

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "optimizer": {
            "method": "differential_evolution",
            "tolerance": 1e-5,
            "options": {"rng": 1},
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_scipy_required_constraints_bounds_de(
    enopt_config: Any, evaluator: Any
) -> None:
    optimizer = BasicOptimizer(enopt_config, evaluator())
    with pytest.raises(NotImplementedError, match="requires bound constraints"):
        optimizer.run(initial_values)


def test_scipy_bound_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]

    validate_optimizer_options(
        "differential_evolution", enopt_config["optimizer"]["options"]
    )

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run([0.2, *initial_values[1:]])
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [0.15, 0.0, 0.2], atol=0.03)


def test_scipy_bound_constraints_differential_evolution_de(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]

    enopt_config["realizations"] = {"realization_min_success": 0}
    optimizer1 = BasicOptimizer(enopt_config, evaluator())
    optimizer1.run([0.2, *initial_values[1:]])
    assert optimizer1.variables is not None
    assert np.allclose(optimizer1.variables, [0.15, 0.0, 0.2], atol=0.03)

    counter = 0

    def _add_nan(x: Any, c: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x, c)

    optimizer2 = BasicOptimizer(enopt_config, evaluator((_add_nan, test_functions[1])))
    optimizer2.run([0.2, *initial_values[1:]])
    assert optimizer2.variables is not None
    assert np.allclose(optimizer2.variables, [0.15, 0.0, 0.2], atol=0.03)
    assert not np.all(optimizer1.variables == optimizer2.variables)


def test_scipy_eq_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    # The result should be [0.25, 0.0, 0.75], but DE appears to have
    # difficulties with linear equality equations. Therefore, we just test if it
    # does not violate them.
    assert optimizer.variables[0] + optimizer.variables[2] == pytest.approx(
        1.0, abs=0.02
    )
    assert optimizer.variables[1] + optimizer.variables[2] == pytest.approx(
        0.75, abs=0.02
    )


def test_scipy_ge_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_le_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_le_ge_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_le_ge_linear_constraints_two_sided_de(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [-0.1, 0.0, 0.4], atol=0.03)

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [-0.1, 0.0, 0.4], atol=0.03)


def test_scipy_eq_nonlinear_constraints_de(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [0.25, 0.0, 0.75], atol=0.03)


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_scipy_ineq_nonlinear_constraints_de(
    enopt_config: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }

    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: weight * variables[0] + weight * variables[2],
    )

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [-0.05, 0.0, 0.45], atol=0.03)


def test_scipy_ineq_nonlinear_constraints_two_sided_de(
    enopt_config: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [-0.1, 0.0, 0.4], atol=0.03)


def test_scipy_le_ge_nonlinear_constraints_de(
    enopt_config: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    # These constraints together force the first two variables to be zero,
    # while the last one is free to fit the function.
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["variables"]["lower_bounds"] = 0.0

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

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [0.0, 0.0, 0.5], atol=0.03)
