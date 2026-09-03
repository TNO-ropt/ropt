"""Default function estimator plugin with mean and standard deviation methods."""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ropt._utils import zero_failures
from ropt.config import FunctionEstimatorConfig
from ropt.context import EnOptContext
from ropt.exceptions import TooFewRealizations
from ropt.function_estimator import FunctionEstimator
from ropt.plugins.function_estimator import FunctionEstimatorPlugin

_MIN_STDDEV_REALIZATIONS: Final = 2

DEFAULT_FUNCTION_ESTIMATOR_METHODS = {"default", "mean", "stddev"}


class DefaultFunctionEstimator(FunctionEstimator):
    """Default estimator providing `mean` and `stddev` methods.

    The method is selected via the `method` field of
    [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig].
    See [Function Estimators](../optimizer_setup/function_estimators.md) for usage.
    """

    def __init__(self, estimator_config: FunctionEstimatorConfig) -> None:
        """Initialize the function estimator.

        Args:
            estimator_config: The function estimator configuration.
        """
        self._estimator_config = estimator_config
        _, _, self._method = self._estimator_config.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "mean"

    def init(self, context: EnOptContext) -> None:  # ruff: ignore[undocumented-public-method]
        self._context = context

    def calculate_function(  # ruff: ignore[undocumented-public-method]
        self,
        functions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._method == "stddev" and self._context.gradient.merge_realizations:
            msg = (
                "The stddev estimator does not support merging "
                "realizations in the gradient."
            )
            raise ValueError(msg)
        estimator_method = self._method
        if estimator_method == "mean":
            return self._calculate_function_mean(functions, weights)
        if estimator_method == "stddev":
            return _calculate_function_stddev(functions, weights)
        msg = f"Function estimator method not supported: {estimator_method}"
        raise ValueError(msg)

    def calculate_gradient(  # ruff: ignore[undocumented-public-method]
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._method == "stddev" and self._context.gradient.merge_realizations:
            msg = (
                "The stddev estimator does not support merging "
                "realizations in the gradient."
            )
            raise ValueError(msg)
        estimator_method = self._method
        if estimator_method == "mean":
            return self._calculate_gradient_mean(
                functions,
                gradient,
                weights,
                merge_realizations=self._context.gradient.merge_realizations,
            )
        if estimator_method == "stddev":
            return _calculate_gradient_stddev(functions, gradient, weights)
        msg = f"Function estimator method not supported: {estimator_method}"
        raise ValueError(msg)

    @staticmethod
    def _calculate_function_mean(
        functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        functions = _contributing(functions, weights)
        return np.dot(functions, weights)  # type: ignore[no-any-return]

    @staticmethod
    def _calculate_gradient_mean(
        _: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
        *,
        merge_realizations: bool = False,
    ) -> NDArray[np.float64]:
        if merge_realizations:
            return gradient
        return np.dot(gradient, weights)  # type: ignore[no-any-return]


def _calculate_function_stddev(
    functions: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    if np.count_nonzero(weights) < _MIN_STDDEV_REALIZATIONS:
        raise TooFewRealizations
    functions = _contributing(functions, weights)
    # Subtracting the equally infinite mean would give NaN, which marks a failure.
    if np.any(np.isinf(functions)):
        return np.array(np.inf, dtype=np.float64)
    *_, stddev = _mean_stddev(functions, weights)
    return stddev


def _calculate_gradient_stddev(
    functions: NDArray[np.float64],
    gradient: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    if np.count_nonzero(weights) < _MIN_STDDEV_REALIZATIONS:
        raise TooFewRealizations
    functions = _contributing(functions, weights)
    if np.any(np.isinf(functions)):
        return np.full(gradient.shape[:-1], np.inf, dtype=np.float64)
    norm, mean, stddev = _mean_stddev(functions, weights)
    mean_gradient = np.dot(gradient, weights)
    return (
        np.zeros(mean_gradient.shape, dtype=np.float64)
        if np.allclose(np.abs(stddev), 0.0)
        else (
            (norm / stddev)
            * (np.dot(gradient, functions * weights) - mean * mean_gradient)
        )
    )


def _contributing(
    functions: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    # A zero weight contributes nothing, but inf * 0 is NaN, so such a value has
    # to be dropped rather than multiplied out.
    return np.where(weights != 0, zero_failures(functions), 0.0)


def _mean_stddev(
    functions: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    norm = float(np.count_nonzero(weights > 0))
    norm /= norm - 1
    mean = np.dot(functions, weights)
    stddev = np.sqrt(norm * np.dot((functions - mean[..., np.newaxis]) ** 2, weights))
    return norm, mean, stddev


class DefaultFunctionEstimatorPlugin(FunctionEstimatorPlugin):
    """Default filter estimator plugin class."""

    @classmethod
    def create(
        cls, estimator_config: FunctionEstimatorConfig
    ) -> DefaultFunctionEstimator:
        """Create a DefaultFunctionEstimator instance.

        Args:
            estimator_config: The function estimator configuration.

        Returns:
            A new `DefaultFunctionEstimator`.
        """
        return DefaultFunctionEstimator(estimator_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in DEFAULT_FUNCTION_ESTIMATOR_METHODS
