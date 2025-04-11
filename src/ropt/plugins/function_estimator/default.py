"""This module implements the default function estimator plugin."""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.enums import OptimizerExitCode
from ropt.exceptions import ConfigError, OptimizationAborted

from .base import FunctionEstimator, FunctionEstimatorPlugin

_MIN_STDDEV_REALIZATIONS: Final = 2


class DefaultFunctionEstimator(FunctionEstimator):
    """The default implementation for function estimation strategies.

    This class provides methods for combining objective function values and
    gradients from an ensemble of realizations into a single representative
    value or gradient. The specific method is configured via the
    [`FunctionEstimatorConfig`][ropt.config.enopt.FunctionEstimatorConfig]
    in the main [`EnOptConfig`][ropt.config.enopt.EnOptConfig].

    **Supported Methods:**

    - `mean` (or `default`):
        Calculates the combined function value as the weighted mean of the
        individual realization function values. The combined gradient is
        calculated as the weighted mean of the individual realization gradients
        (unless `merge_realizations` is true, in which case the pre-merged
        gradient is used directly).

    - `stddev`:
        Calculates the combined function value as the weighted standard
        deviation of the individual realization function values. The combined
        gradient is calculated using the chain rule based on the standard
        deviation formula. This method requires at least two realizations with
        non-zero weights and is incompatible with `merge_realizations=True`
        for gradient calculation.
    """

    def __init__(self, enopt_config: EnOptConfig, estimator_index: int) -> None:
        """Initialize the function estimator object.

        See the
        [ropt.plugins.function_estimator.base.FunctionEstimator][]
        abstract base class.

        # noqa
        """
        self._enopt_config = enopt_config
        self._estimator_config = enopt_config.function_estimators[estimator_index]
        _, _, self._method = self._estimator_config.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "mean"
        if self._method == "stddev" and self._enopt_config.gradient.merge_realizations:
            msg = (
                "The stddev estimator does not support merging "
                "realizations in the gradient."
            )
            raise ConfigError(msg)

    def calculate_function(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate a function from function values for each realization.

        See the
        [ropt.plugins.function_estimator.base.FunctionEstimator][]
        abstract base class.

        # noqa
        """
        estimator_method = self._method
        if estimator_method == "mean":
            return self._calculate_function_mean(functions, weights)
        if estimator_method == "stddev":
            return self._calculate_function_stddev(functions, weights)
        msg = f"Function estimator method not supported: {estimator_method}"
        raise ConfigError(msg)

    def calculate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate a gradient from gradients of the realizations.

        See the
        [ropt.plugins.function_estimator.base.FunctionEstimator][]
        abstract base class.

        # noqa
        """
        estimator_method = self._method
        if estimator_method == "mean":
            return self._calculate_gradient_mean(functions, gradient, weights)
        if estimator_method == "stddev":
            return self._calculate_gradient_stddev(functions, gradient, weights)
        msg = f"Function estimator method not supported: {estimator_method}"
        raise ConfigError(msg)

    @staticmethod
    def _calculate_function_mean(
        functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        functions = np.nan_to_num(functions)
        return np.dot(functions, weights)  # type: ignore[no-any-return]

    def _calculate_gradient_mean(
        self,
        _: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._enopt_config.gradient.merge_realizations:
            return gradient
        return np.dot(gradient, weights)  # type: ignore[no-any-return]

    def _calculate_function_stddev(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if np.count_nonzero(weights) < _MIN_STDDEV_REALIZATIONS:
            raise OptimizationAborted(exit_code=OptimizerExitCode.TOO_FEW_REALIZATIONS)
        functions = np.nan_to_num(functions)
        *_, stddev = self._mean_stddev(functions, weights)
        return stddev

    def _calculate_gradient_stddev(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if np.count_nonzero(weights) < _MIN_STDDEV_REALIZATIONS:
            raise OptimizationAborted(exit_code=OptimizerExitCode.TOO_FEW_REALIZATIONS)
        functions = np.nan_to_num(functions)
        norm, mean, stddev = self._mean_stddev(functions, weights)
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
        self,
        functions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        norm = float(np.count_nonzero(weights > 0))
        norm = norm / (norm - 1)
        mean = np.dot(functions, weights)
        stddev = np.sqrt(
            norm * np.dot((functions - mean[..., np.newaxis]) ** 2, weights)
        )
        return norm, mean, stddev


class DefaultFunctionEstimatorPlugin(FunctionEstimatorPlugin):
    """Default filter estimator plugin class."""

    @classmethod
    def create(
        cls, enopt_config: EnOptConfig, estimator_index: int
    ) -> DefaultFunctionEstimator:
        """Initialize the realization filter plugin.

        See the [ropt.plugins.function_estimator.base.FunctionEstimatorPlugin][]
        abstract base class.

        # noqa
        """
        return DefaultFunctionEstimator(enopt_config, estimator_index)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in {"default", "mean", "stddev"}
