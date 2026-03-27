"""This module implements the default function estimator plugin."""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ropt.config import EnOptConfig, FunctionEstimatorConfig
from ropt.enums import ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.function_estimator import FunctionEstimator

_MIN_STDDEV_REALIZATIONS: Final = 2

DEFAULT_FUNCTION_ESTIMATOR_METHODS = {"default", "mean", "stddev"}


class DefaultFunctionEstimator(FunctionEstimator):
    """The default implementation for function estimation strategies.

    This class provides methods for combining objective function values and
    gradients from an ensemble of realizations into a single representative
    value or gradient. The specific method is configured via the
    [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig] in the main
    [`EnOptConfig`][ropt.config.EnOptConfig].

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

    def __init__(self, estimator_config: FunctionEstimatorConfig) -> None:
        """Initialize the function estimator object.

        See the
        [ropt.function_estimator.FunctionEstimator][]
        abstract base class.

        # noqa
        """
        self._estimator_config = estimator_config
        _, _, self._method = self._estimator_config.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "mean"

    def init(self, enopt_config: EnOptConfig) -> None:
        """Initialize the function estimator object.

        See the
        [ropt.function_estimator.FunctionEstimator][]
        abstract base class.

        # noqa
        """
        self._enopt_config = enopt_config

    def calculate_function(
        self,
        functions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate a function from function values for each realization.

        See the
        [ropt.function_estimator.FunctionEstimator][]
        abstract base class.

        # noqa
        """  # noqa: DOC201, DOC501
        if self._method == "stddev" and self._enopt_config.gradient.merge_realizations:
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

    def calculate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate a gradient from gradients of the realizations.

        See the
        [ropt.function_estimator.FunctionEstimator][]
        abstract base class.

        # noqa
        """  # noqa: DOC201, DOC501
        if self._method == "stddev" and self._enopt_config.gradient.merge_realizations:
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
                merge_realizations=self._enopt_config.gradient.merge_realizations,
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
        raise ComputeStepAborted(exit_code=ExitCode.TOO_FEW_REALIZATIONS)
    functions = np.nan_to_num(functions)
    *_, stddev = _mean_stddev(functions, weights)
    return stddev


def _calculate_gradient_stddev(
    functions: NDArray[np.float64],
    gradient: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    if np.count_nonzero(weights) < _MIN_STDDEV_REALIZATIONS:
        raise ComputeStepAborted(exit_code=ExitCode.TOO_FEW_REALIZATIONS)
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
