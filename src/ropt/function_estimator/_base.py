"""This module defines the abstract base class for function estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class FunctionEstimator(ABC):
    """Abstract Base Class for Function Estimator Implementations.

    This class defines the fundamental interface for all concrete function
    estimator implementations within the `ropt` framework. Function estimators
    provide classes derived from `FunctionEstimator` that encapsulate
    the logic for combining the objective function values and gradients from an
    ensemble of realizations into a single representative value. This aggregated
    value is then used by the core optimization algorithm.

    The core functionality involves combining results using realization weights,
    performed by the `calculate_function` and `calculate_gradient` methods,
    which must be implemented by subclasses.

    Subclasses must implement:

    - `calculate_function`: To combine function values from realizations.
    - `calculate_gradient`: To combine gradient values from realizations.

    """

    @abstractmethod
    def calculate_function(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Combine function values from realizations into an expected value.

        This method takes the function (objective or constraint) values evaluated
        for each realization in the ensemble and combines them into a single
        representative value or vector of values, using the provided realization
        weights.

        Args:
            functions: The function values for each realization.
            weights:   The weight for each realization.

        Returns:
            A scalar or 1D array representing the combined function value(s).
        """

    @abstractmethod
    def calculate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Combine gradients from realizations into an expected gradient.

        This method takes the gradients evaluated for each realization and
        combines them into a single representative gradient vector or matrix,
        using the provided realization weights and potentially the function
        values themselves (e.g., for estimators like standard deviation where
        the chain rule applies).

        Note: Interaction with `merge_realizations`
            The `merge_realizations` flag in the
            [`GradientConfig`][ropt.config.GradientConfig] determines how the
            initial gradient estimate(s) are computed by `ropt` *before* being
            passed to this `calculate_gradient` method.

            - If `False` (default): `ropt` estimates a separate gradient for each
              realization that has a non-zero weight. The implementation
              must then combine these gradients using the provided `weights`.
            - If `True`: `ropt` computes a single, merged gradient estimate by
              treating all perturbations across all realizations collectively.
              The implementation must handle this input appropriately. For simple
              averaging estimators, this might involve returning the input gradient
              unchanged.

            The `merge_realizations=True` option allows gradient estimation even
            with a low number of perturbations (potentially just one) but is
            generally only suitable for estimators performing averaging-like
            operations. Estimator implementations should check this flag during
            initialization (`__init__`) and raise a `ValueError` if
            `merge_realizations=True` is incompatible with the estimator's logic
            (e.g., standard deviation).

        Args:
            functions: The functions for each realization.
            gradient:  The gradient for each realization.
            weights:   The weight of each realization.

        Returns:
            The expected gradients.
        """
