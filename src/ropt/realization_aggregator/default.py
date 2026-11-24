"""Default realization aggregator plugin with mean and standard deviation methods."""

from typing import ClassVar, Final, override

import numpy as np
from numpy.typing import NDArray

from ropt._utils import zero_failures
from ropt.config import RealizationAggregatorConfig
from ropt.exceptions import TooFewRealizations
from ropt.plugins import MethodSpec
from ropt.realization_aggregator import RealizationAggregator

_MIN_STDDEV_REALIZATIONS: Final = 2
_STDDEV_TOLERANCE: Final = 1e-8

DEFAULT_REALIZATION_AGGREGATOR_METHODS = {"default", "mean", "stddev"}


class DefaultRealizationAggregator(RealizationAggregator):
    """Default realization aggregator providing `mean` and `stddev` methods.

    The method is selected via the `method` field of
    [`RealizationAggregatorConfig`][ropt.config.RealizationAggregatorConfig].
    See [Realization Aggregators](../optimizer_setup/realization_aggregators.md) for usage.
    """

    methods: ClassVar[MethodSpec] = DEFAULT_REALIZATION_AGGREGATOR_METHODS

    def __init__(
        self, realization_aggregator_config: RealizationAggregatorConfig
    ) -> None:
        """Initialize the realization aggregator.

        Args:
            realization_aggregator_config: The realization aggregator configuration.
        """
        self._realization_aggregator_config = realization_aggregator_config
        _, _, self._method = (
            self._realization_aggregator_config.method.lower().rpartition("/")
        )
        if self._method == "default":
            self._method = "mean"

    @override
    def init(self, *, merge_realizations: bool) -> None:
        """Record whether the gradient keeps its realizations separate."""
        self._merge_realizations = merge_realizations

    @override
    def aggregate_function(
        self,
        functions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate the function values as a weighted mean or standard deviation.

        Returns:
            The aggregated function value for each objective or constraint.

        Raises:
            ValueError: If the configured method is not supported, or `stddev`
                        is combined with merged realizations.
        """
        if self._method == "stddev" and self._merge_realizations:
            msg = (
                "The stddev realization aggregator does not support merged "
                "realizations in gradient aggregation."
            )
            raise ValueError(msg)
        aggregation_method = self._method
        if aggregation_method == "mean":
            return self._aggregate_function_mean(functions, weights)
        if aggregation_method == "stddev":
            return _aggregate_function_stddev(functions, weights)
        msg = f"Realization aggregator method not supported: {aggregation_method}"
        raise ValueError(msg)

    @override
    def aggregate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate the gradient to match the function aggregation.

        Returns:
            The aggregated gradient.

        Raises:
            ValueError: If the configured method is not supported, or `stddev`
                        is combined with merged realizations.
        """
        if self._method == "stddev" and self._merge_realizations:
            msg = (
                "The stddev realization aggregator does not support merged "
                "realizations in gradient aggregation."
            )
            raise ValueError(msg)
        aggregation_method = self._method
        if aggregation_method == "mean":
            return self._aggregate_gradient_mean(
                functions,
                gradient,
                weights,
                merge_realizations=self._merge_realizations,
            )
        if aggregation_method == "stddev":
            return _aggregate_gradient_stddev(functions, gradient, weights)
        msg = f"Realization aggregator method not supported: {aggregation_method}"
        raise ValueError(msg)

    @staticmethod
    def _aggregate_function_mean(
        functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        functions = _contributing(functions, weights)
        return np.dot(functions, weights)  # type: ignore[no-any-return]

    @staticmethod
    def _aggregate_gradient_mean(
        _: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
        *,
        merge_realizations: bool = False,
    ) -> NDArray[np.float64]:
        if merge_realizations:
            return gradient
        # Preserve the reduction order with realization-first input.
        return np.dot(gradient.T, weights)  # type: ignore[no-any-return]


def _aggregate_function_stddev(
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


def _aggregate_gradient_stddev(
    functions: NDArray[np.float64],
    gradient: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    if np.count_nonzero(weights) < _MIN_STDDEV_REALIZATIONS:
        raise TooFewRealizations
    functions = _contributing(functions, weights)
    if np.any(np.isinf(functions)):
        return np.full(gradient.shape[1:], np.inf, dtype=np.float64)
    norm, mean, stddev = _mean_stddev(functions, weights)
    mean_gradient = np.dot(gradient.T, weights)
    negligible = bool(
        np.all(stddev <= _STDDEV_TOLERANCE * np.max(np.abs(functions), axis=-1))
    )
    return (
        np.zeros(mean_gradient.shape, dtype=np.float64)
        if negligible
        else (
            (norm / stddev)
            * (np.dot(gradient.T, functions * weights) - mean * mean_gradient)
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
