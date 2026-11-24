from functools import partial
from typing import Any, Literal, override

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt import optimize
from ropt._evaluation import EvaluationFunctionContext
from ropt.config._realization_aggregator_config import RealizationAggregatorConfig
from ropt.realization_aggregator import RealizationAggregator
from ropt.realization_aggregator.default import DefaultRealizationAggregator

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


def test_stddev_aggregator_merge_error(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    # Add dummy functions, these will be estimated using stddev.
    test_functions += test_functions

    config["gradient"]["merge_realizations"] = True
    config["objectives"]["weights"].extend([0.75, 0.25])
    config["objectives"]["realization_aggregators"] = [0, 0, 1, 1]
    config["realization_aggregator"] = [{"method": "mean"}, {"method": "stddev"}]
    with pytest.raises(
        ValueError,
        match=(
            "The stddev realization aggregator does not support merged realizations in gradient aggregation"
        ),
    ):
        optimize(config, initial_values, eval_func(test_functions))


def test_mean_stddev_realization_aggregator(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    # Add dummy functions, these will be estimated using stddev.
    test_functions += test_functions

    config["objectives"]["weights"].extend([0.75, 0.25])
    config["objectives"]["realization_aggregators"] = [0, 0, 1, 1]
    config["realization_aggregator"] = [{"method": "mean"}, {"method": "stddev"}]
    result = optimize(config, initial_values, eval_func(test_functions))
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def _compute_distance_squared_stddev(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    target: NDArray[np.float64],
) -> float:
    # To test the stddev aggregator, abuse it to minimize our standard test
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
def test_stddev_realization_aggregator(
    config: Any,
    eval_func: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_compute_distance_squared_stddev, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared_stddev, target=np.array([-1.5, -1.5, 0.5])),
    ]

    config["optimizer"]["evaluation_policy"] = evaluation_policy
    config["realization_aggregator"] = [{"method": "stddev"}]
    result = optimize(config, initial_values, eval_func(functions))
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def _aggregator(method: str) -> DefaultRealizationAggregator:
    aggregator = DefaultRealizationAggregator(
        RealizationAggregatorConfig(method=method)
    )
    aggregator.init(merge_realizations=False)
    return aggregator


def test_mean_realization_aggregator_propagates_an_infinite_realization() -> None:
    aggregator = _aggregator("mean")
    finite = aggregator.aggregate_function(np.array([1.0, 2.0, 3.0]), _EQUAL_WEIGHTS)
    assert finite == pytest.approx(2.0)
    infinite = aggregator.aggregate_function(
        np.array([1.0, np.inf, 3.0]), _EQUAL_WEIGHTS
    )
    assert infinite == np.inf


def test_stddev_realization_aggregator_reports_an_infinite_realization_as_an_infinite_spread() -> (
    None
):
    aggregator = _aggregator("stddev")
    finite = aggregator.aggregate_function(np.array([1.0, 2.0, 3.0]), _EQUAL_WEIGHTS)
    assert finite == pytest.approx(1.0)
    infinite = aggregator.aggregate_function(
        np.array([1.0, np.inf, 3.0]), _EQUAL_WEIGHTS
    )
    assert infinite == np.inf


def test_stddev_gradient_propagates_an_infinite_realization() -> None:
    aggregator = _aggregator("stddev")
    gradient = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    result = aggregator.aggregate_gradient(
        np.array([1.0, np.inf, 3.0]), gradient, _EQUAL_WEIGHTS
    )
    assert result.shape == (2,)
    assert np.all(np.isinf(result))


def test_stddev_gradient_guard_is_scale_invariant() -> None:
    aggregator = _aggregator("stddev")
    # Small enough that an absolute threshold would suppress it.
    functions = np.array([1.0, 1.0 + 1e-7, 1.0 + 2e-7])
    gradient = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    scale = 1e6

    unscaled = aggregator.aggregate_gradient(functions, gradient, _EQUAL_WEIGHTS)
    scaled = aggregator.aggregate_gradient(
        functions / scale, gradient / scale, _EQUAL_WEIGHTS
    )

    assert not np.allclose(unscaled, 0.0)
    assert np.allclose(scaled, unscaled / scale, rtol=1e-5)


def test_stddev_gradient_is_zero_for_identical_realizations() -> None:
    aggregator = _aggregator("stddev")
    gradient = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    result = aggregator.aggregate_gradient(np.zeros(3), gradient, _EQUAL_WEIGHTS)
    assert not np.any(result)


def test_realization_aggregators_ignore_an_infinite_value_carrying_no_weight() -> None:
    # inf * 0 is NaN, so a value excluded by a filter must be dropped, not scaled.
    functions = np.array([1.0, np.inf, 3.0])
    weights = np.array([0.5, 0.0, 0.5])
    mean = _aggregator("mean")
    assert mean.aggregate_function(functions, weights) == pytest.approx(2.0)
    stddev = _aggregator("stddev")
    assert stddev.aggregate_function(functions, weights) == pytest.approx(np.sqrt(2.0))


class CustomRealizationAggregator(RealizationAggregator):
    def __init__(self, _: RealizationAggregatorConfig) -> None:
        pass

    @override
    def init(self, *, merge_realizations: bool) -> None:
        pass

    @override
    def aggregate_function(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.asarray(np.dot(functions, weights) + 1.0)

    @override
    def aggregate_gradient(
        self,
        _: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.asarray(np.dot(gradient.T, weights))


def test_custom_realization_aggregator(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    config["objectives"]["realization_aggregators"] = 0
    config["realization_aggregator"] = [
        CustomRealizationAggregator(RealizationAggregatorConfig(method="custom"))
    ]
    result = optimize(config, initial_values, eval_func(test_functions))
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)
