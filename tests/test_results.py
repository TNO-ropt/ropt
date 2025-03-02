from typing import Any

import numpy as np
import pytest

from ropt.enums import ResultAxis
from ropt.results import FunctionEvaluations, FunctionResults, Functions, Realizations


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
        eval_id=0,
        batch_id=1,
        metadata={},
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


@pytest.mark.parametrize("axis", [ResultAxis.OBJECTIVE, None])
def test_to_dict(function_result: FunctionResults, axis: ResultAxis) -> None:
    names: dict[ResultAxis, tuple[str, ...]] = {ResultAxis.OBJECTIVE: ("f1", "f2")}
    objectives = function_result.evaluations.to_dict(
        "objectives", axis=axis, names=names
    )
    assert np.all(np.equal(objectives["f1"], [0.0, 2.0, 4.0]))
    assert np.all(np.equal(objectives["f2"], [1.0, 3.0, 5.0]))
