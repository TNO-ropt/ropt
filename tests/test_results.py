from typing import Any

import numpy as np
import pytest

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
        batch_id=1,
        metadata={},
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )
