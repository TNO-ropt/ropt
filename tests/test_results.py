from dataclasses import replace
from typing import Any, Dict

import numpy as np
import pytest

from ropt.config.enopt import EnOptConfig
from ropt.enums import ConstraintType
from ropt.results import (
    BoundConstraints,
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Gradients,
    LinearConstraints,
    NonlinearConstraints,
    Realizations,
)


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "names": ["x", "y"],
            "initial_values": [0.0, 0.0],
        },
        "objective_functions": {
            "names": ["f1", "f2"],
            "weights": [0.75, 0.25],
        },
        "realizations": {
            "names": ["r1", "r2", "r3"],
            "weights": [1.0] * 3,
        },
        "gradient": {
            "number_of_perturbations": 5,
        },
    }


@pytest.fixture(name="function_result")
def function_result_fixture(enopt_config: Any) -> FunctionResults:
    config = EnOptConfig.model_validate(enopt_config)
    evaluations = FunctionEvaluations.create(
        config=config,
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    realizations = Realizations(
        objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
        failed_realizations=np.zeros(3, dtype=np.bool_),
    )
    functions = Functions.create(
        config=config, weighted_objective=np.array(1.0), objectives=np.array([1.0, 2.0])
    )
    return FunctionResults(
        result_id=0,
        batch_id=1,
        metadata={},
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


@pytest.fixture(name="gradient_result")
def gradient_result_fixture() -> GradientResults:
    evaluations = GradientEvaluations(
        variables=np.array([1.0, 2.0]),
        perturbed_variables=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        perturbed_objectives=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
    )
    gradients = Gradients(
        weighted_objective=np.array([1.0, 2.0]),
        objectives=np.arange(4, dtype=np.float64).reshape((2, 2)),
    )
    return GradientResults(
        result_id=0,
        batch_id=1,
        metadata={},
        evaluations=evaluations,
        realizations=Realizations(
            failed_realizations=np.zeros(36, dtype=np.bool_),
        ),
        gradients=gradients,
    )


def test_get_axis_names_error(function_result: FunctionResults) -> None:
    with pytest.raises(ValueError, match="Unknown field name: foo"):
        function_result.evaluations.get_axis_names("foo")


def test_scaling_evaluations_functions(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["variables"]["scales"] = [2.0, 2.0]
    enopt_config["objective_functions"]["scales"] = [2.0, 2.0]
    config = EnOptConfig.model_validate(enopt_config)
    evaluations = FunctionEvaluations.create(
        config=config,
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    function_result = replace(function_result, evaluations=evaluations)
    assert config.variables.scales is not None
    assert function_result.evaluations.unscaled_variables is not None
    assert np.allclose(
        function_result.evaluations.unscaled_variables,
        function_result.evaluations.variables * config.variables.scales,
    )
    assert function_result.evaluations.scaled_objectives is not None
    assert np.allclose(
        function_result.evaluations.scaled_objectives,
        function_result.evaluations.objectives / config.objective_functions.scales,
    )


def test_scaling_evaluations_functions_auto(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["objective_functions"]["scales"] = [2.0, 2.0]
    config = EnOptConfig.model_validate(enopt_config)
    objective_auto_scales = np.array([3.0, 3.0])
    evaluations = FunctionEvaluations.create(
        config=config,
        objective_auto_scales=objective_auto_scales,
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    function_result = replace(function_result, evaluations=evaluations)
    assert function_result.evaluations.scaled_objectives is not None
    assert np.allclose(
        function_result.evaluations.scaled_objectives,
        function_result.evaluations.objectives
        / config.objective_functions.scales
        / objective_auto_scales,
    )


def test_scaling_functions(enopt_config: Any, function_result: FunctionResults) -> None:
    enopt_config["objective_functions"]["scales"] = [2.0, 2.0]
    config = EnOptConfig.model_validate(enopt_config)
    functions = Functions.create(
        config=config,
        weighted_objective=np.array(1.0),
        objectives=np.array([1.0, 2.0]),
    )
    function_result = replace(function_result, functions=functions)
    assert function_result.functions is not None
    assert function_result.functions.scaled_objectives is not None
    assert np.allclose(
        function_result.functions.scaled_objectives,
        function_result.functions.objectives / config.objective_functions.scales,
    )


def test_scaling_functions_auto(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["objective_functions"]["scales"] = [2.0, 2.0]
    config = EnOptConfig.model_validate(enopt_config)
    objective_auto_scales = np.array([3.0, 3.0])
    functions = Functions.create(
        config=config,
        objective_auto_scales=objective_auto_scales,
        weighted_objective=np.array(1.0),
        objectives=np.array([1.0, 2.0]),
    )
    function_result = replace(function_result, functions=functions)
    assert function_result.functions is not None
    assert function_result.functions.scaled_objectives is not None
    assert np.allclose(
        function_result.functions.scaled_objectives,
        function_result.functions.objectives
        / config.objective_functions.scales
        / objective_auto_scales,
    )


def test_bound_constraint_results(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["variables"]["lower_bounds"] = [1.2, 1.4]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.5]
    config = EnOptConfig.model_validate(enopt_config)
    assert function_result.functions is not None
    constraints = BoundConstraints.create(config, function_result.evaluations)
    assert constraints is not None
    assert constraints.lower_values is not None
    assert constraints.upper_values is not None
    assert constraints.lower_violations is not None
    assert constraints.upper_violations is not None
    assert np.allclose(constraints.lower_values, [-0.2, 0.6])
    assert np.allclose(constraints.upper_values, [0.0, 0.5])
    assert np.allclose(constraints.lower_violations, [0.2, 0.0])
    assert np.allclose(constraints.upper_violations, [0.0, 0.5])


def test_linear_constraint_results(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1.0, 0.0], [0.0, 1.0]],
        "rhs_values": [0.0, 1.0],
        "types": [ConstraintType.LE, ConstraintType.GE],
    }
    config = EnOptConfig.model_validate(enopt_config)
    assert function_result.functions is not None
    constraints = LinearConstraints.create(config, function_result.evaluations)
    assert constraints is not None
    assert constraints.values is not None  # noqa: PD011
    assert constraints.violations is not None
    assert np.allclose(constraints.values, [1.0, 1.0])
    assert np.allclose(constraints.violations, [1.0, 0.0])


def test_nonlinear_constraint_results(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": [0.0, 1.0, 0.0],
        "types": [ConstraintType.LE, ConstraintType.GE, ConstraintType.EQ],
    }
    config = EnOptConfig.model_validate(enopt_config)
    assert function_result.functions is not None
    function_result.functions.constraints = np.array([1.0, 1.0, -1.0])
    constraints = NonlinearConstraints.create(config, function_result.functions, None)
    assert constraints is not None
    assert constraints.values is not None  # noqa: PD011
    assert constraints.violations is not None
    assert np.allclose(constraints.values, [1.0, 0.0, -1.0])
    assert np.allclose(constraints.violations, [1.0, 0.0, 1.0])
