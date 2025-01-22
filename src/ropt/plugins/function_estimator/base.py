"""This module defines the abstract base classes for function estimators.

Function estimators can be added via the plugin mechanism to implement
additional ways to functions and gradient ensembles. Any object that adheres to
the
[`FunctionEstimator`][ropt.plugins.function_estimator.base.FunctionEstimator]
base class may be installed as a plugin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class FunctionEstimator(ABC):
    """Abstract base class for function estimators."""

    @abstractmethod
    def __init__(self, enopt_config: EnOptConfig, estimator_index: int) -> None:
        """Initialize the function estimator object.

        Args:
            enopt_config:    The configuration of the optimizer.
            estimator_index: The index of the estimator to use.
        """

    @abstractmethod
    def calculate_function(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Combine functions from realizations into an expected function.

        Tip: Calculation from merged realizations
           Normally the gradient is calculated for each realization separately
           and then combined into an overall gradient with `calculate_gradient`
           method. The `merge_realizations` flag in the ensemble optimizer
           configuration directs the optimizer to calculate the overall gradient
           from all realizations directly. This yields a reasonable estimation
           if the function estimator is an averaging operation, but may not be
           appropriate in other cases.

           At initialization, the `merge_realizations` flag should be checked,
           and if necessary a `ConfigError` with an appropriate message should
           be raised.

        Args:
            functions: The functions for each realization.
            weights:   The weight of each realization.

        Returns:
            The expected function values.
        """

    @abstractmethod
    def calculate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Combine gradients from realizations into an expected gradient.

        Args:
            functions: The functions for each realization.
            gradient:  The gradient for each realization.
            weights:   The weight of each realization.

        Returns:
            The expected gradients.
        """


class FunctionEstimatorPlugin(Plugin):
    """The function estimator plugin base class."""

    @abstractmethod
    def create(
        self, enopt_config: EnOptConfig, estimator_index: int
    ) -> FunctionEstimator:
        """Initialize the function estimator object.

        Args:
            enopt_config:    The configuration of the optimizer.
            estimator_index: The index of the estimator to use.
        """
