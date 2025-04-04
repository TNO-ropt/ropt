from typing import Any

import numpy as np
import pytest

from ropt.plan import BasicOptimizer
from ropt.plugins import PluginManager

pytestmark = [pytest.mark.slow]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "method": "differential_evolution",
            "tolerance": 1e-4,
            "max_iterations": 20,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_scipy_required_constraints_bounds_de(
    enopt_config: Any, evaluator: Any
) -> None:
    with pytest.raises(NotImplementedError, match="requires bound constraints"):
        BasicOptimizer(enopt_config, evaluator()).run()


def test_scipy_bound_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    enopt_config["variables"]["initial_values"][0] = 0.2
    enopt_config["optimizer"]["options"] = {"seed": 1}

    plugin_manager = PluginManager()
    plugin = plugin_manager.get_plugin("optimizer", "differential_evolution")
    plugin.validate_options(
        "differential_evolution", enopt_config["optimizer"]["options"]
    )

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=0.02)


def test_scipy_bound_constraints_differential_evolution_de(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    enopt_config["variables"]["initial_values"][0] = 0.2
    enopt_config["optimizer"]["options"] = {"seed": 1}

    enopt_config["realizations"] = {"realization_min_success": 0}
    variables1 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables1 is not None
    assert np.allclose(variables1, [0.15, 0.0, 0.2], atol=0.025)

    counter = 0

    def _add_nan(x: Any, c: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x, c)

    variables2 = (
        BasicOptimizer(enopt_config, evaluator((_add_nan, test_functions[1])))
        .run()
        .variables
    )
    assert variables2 is not None
    assert np.allclose(variables2, [0.15, 0.0, 0.2], atol=0.025)
    assert not np.all(variables1 == variables2)


def test_scipy_eq_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    # The result should be [0.25, 0.0, 0.75], but DE appears to have
    # difficulties with linear equality equations. Therefore, we just test if it
    # does not violate them.
    assert variables[0] + variables[2] == pytest.approx(1.0, abs=0.02)
    assert variables[1] + variables[2] == pytest.approx(0.75, abs=0.02)


def test_scipy_ge_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }
    enopt_config["optimizer"]["options"] = {"seed": 1}

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_scipy_le_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    enopt_config["optimizer"]["options"] = {"seed": 1}

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_scipy_le_ge_linear_constraints_de(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }
    enopt_config["optimizer"]["options"] = {"seed": 1}

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


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
    enopt_config["optimizer"]["options"] = {"seed": 1}

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


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

    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


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
    enopt_config["optimizer"]["options"] = {"seed": 1}

    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: weight * variables[0] + weight * variables[2],
    )

    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


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
    enopt_config["optimizer"]["options"] = {"seed": 1}
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


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

    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)
