from functools import partial
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.workflow import BasicOptimizer

initial_values = 3 * [0]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-4,
            "max_functions": 10,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "number_of_perturbations": 5,
        },
        "realizations": {"weights": 5 * [1.0]},
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
    }


def test_stddev_function_estimator_merge_error(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    # Add dummy functions, these will be estimated using stddev.
    test_functions = test_functions + test_functions

    enopt_config["gradient"]["merge_realizations"] = True
    enopt_config["objectives"]["weights"].extend([0.75, 0.25])
    enopt_config["objectives"]["function_estimators"] = [0, 0, 1, 1]
    enopt_config["function_estimators"] = [{"method": "mean"}, {"method": "stddev"}]
    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    with pytest.raises(
        ValueError,
        match=(
            "The stddev estimator does not support merging realizations in the gradient"
        ),
    ):
        optimizer.run(initial_values)


def test_mean_stddev_function_estimator(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    # Add dummy functions, these will be estimated using stddev.
    test_functions = test_functions + test_functions

    enopt_config["objectives"]["weights"].extend([0.75, 0.25])
    enopt_config["objectives"]["function_estimators"] = [0, 0, 1, 1]
    enopt_config["function_estimators"] = [{"method": "mean"}, {"method": "stddev"}]
    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [0.0, 0.0, 0.5], atol=0.02)


def _compute_distance_squared_stddev(
    variables: NDArray[np.float64], realization: int, target: NDArray[np.float64]
) -> float:
    # To test the stddev estimator, abuse it to minimize our standard test
    # function, the squared distance between variables and targets. Do the
    # following:
    # - Set one realization to the sum of the squared differences
    # - Set one realization to zero
    # - Set a third equal to the negative of the first
    # The mean of these three realizations is zero, and their standard deviation
    # is equal to the squared distance. Hence, using the standard deviation
    # objective function will optimize the squared distance.
    result: float = ((variables - target) ** 2).sum()
    if realization in [0, 1]:
        result = -result
    elif realization == 2:
        result = 0.0
    return result


@pytest.mark.parametrize("evaluation_policy", ["separate", "auto"])
def test_stddev_function_estimator(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_compute_distance_squared_stddev, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared_stddev, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy
    enopt_config["function_estimators"] = [{"method": "stddev"}]
    optimizer = BasicOptimizer(enopt_config, evaluator(functions))
    optimizer.run(initial_values)
    assert optimizer.variables is not None
    assert np.allclose(optimizer.variables, [0.0, 0.0, 0.5], atol=0.02)
