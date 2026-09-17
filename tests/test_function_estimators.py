from functools import partial
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.config._function_estimator_config import FunctionEstimatorConfig
from ropt.function_estimator import FunctionEstimator
from ropt.function_estimator.default import DefaultFunctionEstimator
from ropt.simple import optimize

initial_values = 3 * [0]

_EQUAL_WEIGHTS = np.array(3 * [1.0 / 3.0])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 10,
        },
        "backend": {
            "convergence_tolerance": 1e-4,
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
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    # Add dummy functions, these will be estimated using stddev.
    test_functions += test_functions

    config["gradient"]["merge_realizations"] = True
    config["objectives"]["weights"].extend([0.75, 0.25])
    config["objectives"]["function_estimators"] = [0, 0, 1, 1]
    config["function_estimators"] = [{"method": "mean"}, {"method": "stddev"}]
    with pytest.raises(
        ValueError,
        match=(
            "The stddev estimator does not support merging realizations in the gradient"
        ),
    ):
        optimize(config, initial_values, eval_func(test_functions))


def test_mean_stddev_function_estimator(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    # Add dummy functions, these will be estimated using stddev.
    test_functions += test_functions

    config["objectives"]["weights"].extend([0.75, 0.25])
    config["objectives"]["function_estimators"] = [0, 0, 1, 1]
    config["function_estimators"] = [{"method": "mean"}, {"method": "stddev"}]
    result = optimize(config, initial_values, eval_func(test_functions))
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def _compute_distance_squared_stddev(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    target: NDArray[np.float64],
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
    if context.realization in {0, 1}:
        result = -result
    elif context.realization == 2:
        result = 0.0
    return result


@pytest.mark.parametrize("evaluation_policy", ["separate", "auto"])
def test_stddev_function_estimator(
    config: Any,
    eval_func: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_compute_distance_squared_stddev, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared_stddev, target=np.array([-1.5, -1.5, 0.5])),
    ]

    config["gradient"]["evaluation_policy"] = evaluation_policy
    config["function_estimators"] = [{"method": "stddev"}]
    result = optimize(config, initial_values, eval_func(functions))
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def _estimator(method: str) -> DefaultFunctionEstimator:
    estimator = DefaultFunctionEstimator(FunctionEstimatorConfig(method=method))
    estimator.init(merge_realizations=False)
    return estimator


def test_mean_estimator_propagates_an_infinite_realization() -> None:
    estimator = _estimator("mean")
    finite = estimator.calculate_function(np.array([1.0, 2.0, 3.0]), _EQUAL_WEIGHTS)
    assert finite == pytest.approx(2.0)
    infinite = estimator.calculate_function(
        np.array([1.0, np.inf, 3.0]), _EQUAL_WEIGHTS
    )
    assert infinite == np.inf


def test_stddev_estimator_reports_an_infinite_realization_as_an_infinite_spread() -> (
    None
):
    estimator = _estimator("stddev")
    finite = estimator.calculate_function(np.array([1.0, 2.0, 3.0]), _EQUAL_WEIGHTS)
    assert finite == pytest.approx(1.0)
    infinite = estimator.calculate_function(
        np.array([1.0, np.inf, 3.0]), _EQUAL_WEIGHTS
    )
    assert infinite == np.inf


def test_stddev_gradient_propagates_an_infinite_realization() -> None:
    estimator = _estimator("stddev")
    gradient = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = estimator.calculate_gradient(
        np.array([1.0, np.inf, 3.0]), gradient, _EQUAL_WEIGHTS
    )
    assert result.shape == (2,)
    assert np.all(np.isinf(result))


def test_stddev_gradient_guard_is_scale_invariant() -> None:
    estimator = _estimator("stddev")
    # Small enough that an absolute threshold would suppress it.
    functions = np.array([1.0, 1.0 + 1e-7, 1.0 + 2e-7])
    gradient = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scale = 1e6

    unscaled = estimator.calculate_gradient(functions, gradient, _EQUAL_WEIGHTS)
    scaled = estimator.calculate_gradient(
        functions / scale, gradient / scale, _EQUAL_WEIGHTS
    )

    assert not np.allclose(unscaled, 0.0)
    assert np.allclose(scaled, unscaled / scale, rtol=1e-5)


def test_stddev_gradient_is_zero_for_identical_realizations() -> None:
    estimator = _estimator("stddev")
    gradient = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = estimator.calculate_gradient(np.zeros(3), gradient, _EQUAL_WEIGHTS)
    assert not np.any(result)


def test_estimators_ignore_an_infinite_value_carrying_no_weight() -> None:
    # inf * 0 is NaN, so a value excluded by a filter must be dropped, not scaled.
    functions = np.array([1.0, np.inf, 3.0])
    weights = np.array([0.5, 0.0, 0.5])
    mean = _estimator("mean")
    assert mean.calculate_function(functions, weights) == pytest.approx(2.0)
    stddev = _estimator("stddev")
    assert stddev.calculate_function(functions, weights) == pytest.approx(np.sqrt(2.0))


class CustomFunctionEstimator(FunctionEstimator):
    def __init__(self, _: FunctionEstimatorConfig) -> None:
        pass

    def init(self, *, merge_realizations: bool) -> None:
        pass

    def calculate_function(  # ruff: ignore[no-self-use]
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.asarray(np.dot(functions, weights) + 1.0)

    def calculate_gradient(  # ruff: ignore[no-self-use]
        self,
        _: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.asarray(np.dot(gradient, weights))


def test_custom_function_estimator(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    config["objectives"]["function_estimators"] = 0
    config["function_estimators"] = [
        CustomFunctionEstimator(FunctionEstimatorConfig(method="custom"))
    ]
    result = optimize(config, initial_values, eval_func(test_functions))
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)
