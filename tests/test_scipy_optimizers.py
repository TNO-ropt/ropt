from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.enums import ConstraintType, EventType
from ropt.events import OptimizationEvent
from ropt.plan import BasicOptimizationPlan
from ropt.plugins.optimizer.scipy import (
    _CONSTRAINT_REQUIRES_BOUNDS,
    _CONSTRAINT_SUPPORT_BOUNDS,
    _CONSTRAINT_SUPPORT_LINEAR_EQ,
    _CONSTRAINT_SUPPORT_LINEAR_INEQ,
    _CONSTRAINT_SUPPORT_NONLINEAR_EQ,
    _CONSTRAINT_SUPPORT_NONLINEAR_INEQ,
    _SUPPORTED_METHODS,
)


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "tolerance": 1e-4,
            "max_iterations": 50,
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


@pytest.mark.parametrize(
    "method", sorted(_SUPPORTED_METHODS - _CONSTRAINT_REQUIRES_BOUNDS)
)
def test_scipy_unconstrained(enopt_config: Any, method: str, evaluator: Any) -> None:
    enopt_config["optimizer"]["method"] = method

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    assert variables is not None
    # Some methods are supported, but not reliable in this test.
    if method != "newton-cg":
        assert np.allclose(variables, [0, 0, 0.5], atol=0.02)


@pytest.mark.parametrize(
    "method", sorted(_SUPPORTED_METHODS - _CONSTRAINT_SUPPORT_BOUNDS)
)
def test_scipy_unsupported_constraints_bounds(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["variables"]["lower_bounds"] = [0.1, -np.inf, -np.inf]
    with pytest.raises(NotImplementedError, match="does not support bound constraints"):
        BasicOptimizationPlan(enopt_config, evaluator()).run()


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_REQUIRES_BOUNDS))
def test_scipy_required_constraints_bounds(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    with pytest.raises(NotImplementedError, match="requires bound constraints"):
        BasicOptimizationPlan(enopt_config, evaluator()).run()


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED_METHODS - _CONSTRAINT_SUPPORT_LINEAR_EQ),
)
def test_scipy_unsupported_constraints_linear_eq(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1.0, 0.0, 1.0]],
        "rhs_values": 1.0,
        "types": [ConstraintType.EQ],
    }
    with pytest.raises(
        NotImplementedError, match="does not support linear equality constraints"
    ):
        BasicOptimizationPlan(enopt_config, evaluator()).run()


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED_METHODS - _CONSTRAINT_SUPPORT_LINEAR_INEQ),
)
@pytest.mark.parametrize("bound_type", [ConstraintType.LE, ConstraintType.GE])
def test_scipy_unsupported_constraints_linear_ineq(
    enopt_config: Any,
    method: str,
    bound_type: ConstraintType,
    evaluator: Any,
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1.0, 0.0, 1.0]],
        "rhs_values": 0.4 if bound_type == ConstraintType.LE else -0.4,
        "types": [bound_type],
    }
    with pytest.raises(
        NotImplementedError, match="does not support linear inequality constraints"
    ):
        BasicOptimizationPlan(enopt_config, evaluator()).run()


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED_METHODS - _CONSTRAINT_SUPPORT_NONLINEAR_EQ),
)
def test_scipy_unsupported_constraints_nonlinear_eq(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 1.0,
        "types": [ConstraintType.EQ],
    }

    with pytest.raises(
        NotImplementedError, match="does not support non-linear equality constraints"
    ):
        BasicOptimizationPlan(enopt_config, evaluator()).run()


@pytest.mark.parametrize(
    "method",
    sorted(_SUPPORTED_METHODS - _CONSTRAINT_SUPPORT_NONLINEAR_INEQ),
)
@pytest.mark.parametrize("bound_type", [ConstraintType.LE, ConstraintType.GE])
def test_scipy_unsupported_constraints_nonlinear_ineq(
    enopt_config: Any,
    method: str,
    bound_type: ConstraintType,
    evaluator: Any,
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4 if bound_type == ConstraintType.LE else -0.0,
        "types": [bound_type],
    }
    with pytest.raises(
        NotImplementedError, match="does not support non-linear inequality constraints"
    ):
        BasicOptimizationPlan(enopt_config, evaluator()).run()


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_SUPPORT_BOUNDS))
def test_scipy_bound_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    enopt_config["variables"]["initial_values"][0] = 0.2

    if method == "differential_evolution":
        enopt_config["optimizer"]["options"] = {"seed": 123}
    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    assert variables is not None
    if method != "differential_evolution":
        assert np.allclose(variables, [0.15, 0.0, 0.2], atol=0.02)


def test_scipy_bound_constraints_differential_evolution(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["optimizer"]["method"] = "differential_evolution"
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    enopt_config["variables"]["initial_values"][0] = 0.2
    enopt_config["optimizer"]["options"] = {"seed": 123}
    enopt_config["realizations"] = {"realization_min_success": 0}
    variables1 = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
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
        BasicOptimizationPlan(enopt_config, evaluator((_add_nan, test_functions[1])))
        .run()
        .variables
    )
    assert variables2 is not None
    assert np.allclose(variables2, [0.15, 0.0, 0.2], atol=0.025)
    assert not np.all(variables1 == variables2)


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_SUPPORT_LINEAR_EQ))
def test_scipy_eq_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _CONSTRAINT_SUPPORT_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "rhs_values": [1.0, 0.75],
        "types": [ConstraintType.EQ, ConstraintType.EQ],
    }

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    if method != "differential_evolution":
        assert variables is not None
        assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_SUPPORT_LINEAR_INEQ))
def test_scipy_ge_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _CONSTRAINT_SUPPORT_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "rhs_values": -0.4,
        "types": [ConstraintType.GE],
    }

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    if method != "differential_evolution":
        assert variables is not None
        assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_SUPPORT_LINEAR_INEQ))
def test_scipy_le_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _CONSTRAINT_SUPPORT_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "rhs_values": 0.4,
        "types": [ConstraintType.LE],
    }

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    if method != "differential_evolution":
        assert variables is not None
        assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_SUPPORT_LINEAR_INEQ))
def test_scipy_le_ge_linear_constraints(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    if method in _CONSTRAINT_SUPPORT_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "rhs_values": [0.4, -0.4],
        "types": [ConstraintType.LE, ConstraintType.GE],
    }

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    if method != "differential_evolution":
        assert variables is not None
        assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_SUPPORT_NONLINEAR_EQ))
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
        "rhs_values": 1.0,
        "types": [ConstraintType.EQ],
    }

    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0] + variables[2]),
    )

    variables = (
        BasicOptimizationPlan(enopt_config, evaluator(test_functions)).run().variables
    )
    if method != "differential_evolution":
        assert variables is not None
        assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("method", sorted(_CONSTRAINT_SUPPORT_NONLINEAR_INEQ))
@pytest.mark.parametrize("bound_type", [ConstraintType.LE, ConstraintType.GE])
def test_scipy_ineq_nonlinear_constraints(
    enopt_config: Any,
    method: str,
    bound_type: ConstraintType,
    evaluator: Any,
    test_functions: Any,
) -> None:
    if method in _CONSTRAINT_SUPPORT_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["method"] = method
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4 if bound_type == ConstraintType.LE else -0.4,
        "types": [bound_type],
    }

    weight = 1.0 if bound_type == ConstraintType.LE else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: cast(
            NDArray[np.float64], weight * variables[0] + weight * variables[2]
        ),
    )

    variables = (
        BasicOptimizationPlan(enopt_config, evaluator(test_functions)).run().variables
    )
    if method != "differential_evolution":
        assert variables is not None
        assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize(
    "method",
    sorted(_CONSTRAINT_SUPPORT_NONLINEAR_INEQ & _CONSTRAINT_SUPPORT_NONLINEAR_EQ),
)
def test_scipy_le_ge_nonlinear_constraints(
    enopt_config: Any,
    method: str,
    evaluator: Any,
    test_functions: Any,
) -> None:
    # These constraints together force the first two variables to be zero,
    # while the last one is free to fit the function.
    if method in _CONSTRAINT_SUPPORT_BOUNDS:
        enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
        enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["variables"]["lower_bounds"] = 0.0
    enopt_config["optimizer"]["method"] = method

    enopt_config["nonlinear_constraints"] = {
        "rhs_values": [0.4, 0.0],
        "types": [ConstraintType.GE, ConstraintType.EQ],
    }

    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0] + variables[2]),
    )
    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0] - variables[1]),
    )

    variables = (
        BasicOptimizationPlan(enopt_config, evaluator(test_functions)).run().variables
    )
    if method != "differential_evolution":
        assert variables is not None
        assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_scipy_options(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["method"] = "Nelder-Mead"
    enopt_config["optimizer"]["options"] = {"maxfev": 10}

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert pytest.approx(variables[2], abs=0.025) != 0.5


@pytest.mark.parametrize(
    "method", sorted(_SUPPORTED_METHODS - _CONSTRAINT_REQUIRES_BOUNDS)
)
def test_scipy_split_evaluations(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = method
    enopt_config["optimizer"]["split_evaluations"] = True

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    assert variables is not None
    # Some methods are supported, but not reliable in this test.
    if method != "newton-cg":
        assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

    enopt_config["optimizer"]["method"] = method
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["split_evaluations"] = True

    variables = BasicOptimizationPlan(enopt_config, evaluator()).run().variables
    assert variables is not None
    # Some methods are supported, but not reliable in this test.
    if method != "newton-cg":
        assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize("speculative", [True, False])
def test_scipy_speculative(
    enopt_config: Any, evaluator: Any, speculative: bool
) -> None:
    enopt_config["optimizer"]["method"] = "slsqp"
    enopt_config["optimizer"]["speculative"] = speculative

    def _observer(event: OptimizationEvent) -> None:
        assert event.results is not None
        assert len(event.results) == 2 if speculative else 1

    variables = (
        BasicOptimizationPlan(enopt_config, evaluator())
        .add_observer(EventType.FINISHED_EVALUATION, _observer)
        .run()
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_scipy_output_dir(tmp_path: Path, enopt_config: Any, evaluator: Any) -> None:
    output_dir = tmp_path / "outputdir"
    output_dir.mkdir()
    enopt_config["optimizer"]["output_dir"] = output_dir
    enopt_config["optimizer"]["method"] = "slsqp"
    enopt_config["optimizer"]["max_functions"] = 1

    BasicOptimizationPlan(enopt_config, evaluator()).run()
    assert (output_dir / "optimizer_output.txt").exists()

    BasicOptimizationPlan(enopt_config, evaluator()).run()
    assert (output_dir / "optimizer_output-001.txt").exists()

    BasicOptimizationPlan(enopt_config, evaluator()).run()
    assert (output_dir / "optimizer_output-002.txt").exists()
