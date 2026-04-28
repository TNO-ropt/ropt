"""Default function estimator plugin with mean and standard deviation methods.

This module provides the built-in implementation of the
[`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] interface,
offering two aggregation strategies: weighted mean (the default) and weighted
standard deviation. These are the primary aggregation methods used in most
ensemble-based optimization workflows.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ropt.config import FunctionEstimatorConfig
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import Abort
from ropt.function_estimator import FunctionEstimator

_MIN_STDDEV_REALIZATIONS: Final = 2

DEFAULT_FUNCTION_ESTIMATOR_METHODS = {"default", "mean", "stddev"}


class DefaultFunctionEstimator(FunctionEstimator):
    """Default implementation of function estimator with mean and stddev methods.

    Implements the [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator]
    interface to provide two standard aggregation strategies for combining
    objective/constraint function values and gradients from multiple realizations.

    The specific method is selected via the `method` field of the
    [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig], which is
    passed to the parent class during initialization.

    **Supported Aggregation Methods**

    - **`mean` (or `default`)**:
       Computes the weighted average of realization function values and
       gradients. For functions, returns: `sum(functions[i] * weights[i])`. For
       gradients, returns the weighted average of per-realization gradients, or
       the pre-merged gradient if `merge_realizations=True` in the gradient
       configuration.
    - **`stddev`**:
       Computes the weighted standard deviation of realization function values.
       For functions, returns the sample standard deviation weighted by
       realization weights. For gradients, applies the chain rule to combine
       function values and per-realization gradients.
        - Requires at least two realizations with non-zero weights.
        - Incompatible with `merge_realizations=True`; raises `ValueError` during
          `init` if this setting is detected.
    """

    def __init__(self, estimator_config: FunctionEstimatorConfig) -> None:  # noqa: D107
        self._estimator_config = estimator_config
        _, _, self._method = self._estimator_config.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "mean"

    def init(self, context: EnOptContext) -> None:  # noqa: D102
        self._context = context

    def calculate_function(  # noqa: D102
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

    def calculate_gradient(  # noqa: D102
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
        functions = np.nan_to_num(functions)
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
        raise Abort(exit_code=ExitCode.TOO_FEW_REALIZATIONS)
    functions = np.nan_to_num(functions)
    *_, stddev = _mean_stddev(functions, weights)
    return stddev


def _calculate_gradient_stddev(
    functions: NDArray[np.float64],
    gradient: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    if np.count_nonzero(weights) < _MIN_STDDEV_REALIZATIONS:
        raise Abort(exit_code=ExitCode.TOO_FEW_REALIZATIONS)
    functions = np.nan_to_num(functions)
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


def _mean_stddev(
    functions: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    norm = float(np.count_nonzero(weights > 0))
    norm /= norm - 1
    mean = np.dot(functions, weights)
    stddev = np.sqrt(norm * np.dot((functions - mean[..., np.newaxis]) ** 2, weights))
    return norm, mean, stddev
