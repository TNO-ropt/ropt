from typing import Any

import numpy as np
import pytest

from ropt.config.enopt import EnOptConfig
from ropt.enums import ConstraintType, ResultAxis
from ropt.results import (
    BoundConstraints,
    FunctionEvaluations,
    FunctionResults,
    Functions,
    LinearConstraints,
    Realizations,
)


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0],
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "realizations": {
            "weights": [1.0] * 3,
        },
        "gradient": {
            "number_of_perturbations": 5,
        },
    }


@pytest.fixture(name="function_result")
def function_result_fixture() -> FunctionResults:
    evaluations = FunctionEvaluations.create(
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    realizations = Realizations(
        objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
        failed_realizations=np.zeros(3, dtype=np.bool_),
    )
    functions = Functions.create(
        weighted_objective=np.array(1.0), objectives=np.array([1.0, 2.0])
    )
    return FunctionResults(
        plan_id=(0,),
        batch_id=1,
        metadata={},
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
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
    assert constraints.values is not None
    assert constraints.violations is not None
    assert np.allclose(constraints.values, [1.0, 1.0])
    assert np.allclose(constraints.violations, [1.0, 0.0])


@pytest.mark.parametrize("axis", [ResultAxis.OBJECTIVE, None])
def test_to_dict(function_result: FunctionResults, axis: ResultAxis) -> None:
    names: dict[ResultAxis, tuple[str, ...]] = {ResultAxis.OBJECTIVE: ("f1", "f2")}
    objectives = function_result.evaluations.to_dict(
        "objectives", axis=axis, names=names
    )
    assert np.all(np.equal(objectives["f1"], [0.0, 2.0, 4.0]))
    assert np.all(np.equal(objectives["f2"], [1.0, 3.0, 5.0]))
