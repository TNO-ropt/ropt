"""Abstract base class for function estimator implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import FunctionEstimatorConfig
    from ropt.context import EnOptContext


class FunctionEstimator(ABC):
    """Abstract base class for function estimator implementations.

    Subclasses must implement four methods:

    1. `__init__` — store configuration; defer heavy work to `init`.
    2. `init` — called once with the full optimization context; validate
       settings and pre-compute state here.
    3. `calculate_function` — aggregate per-realization function values.
    4. `calculate_gradient` — aggregate per-realization gradients.

    See [Function Estimators](../usage/function_estimators.md) for examples
    and further guidance.
    """

    @abstractmethod
    def __init__(self, estimator_config: FunctionEstimatorConfig) -> None:
        """Create a new function estimator instance.

        Store the configuration; keep initialization lightweight.
        Context-dependent setup belongs in `init`.

        Args:
            estimator_config: The estimator configuration.
        """

    @abstractmethod
    def init(self, context: EnOptContext) -> None:
        """Finalize initialization with the optimization context.

        Called once after configuration is finalized. Use for validation
        (e.g., compatibility with `merge_realizations`) and precomputation.

        Args:
            context: The optimization context.
        """

    @abstractmethod
    def calculate_function(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Aggregate function values across realizations.

        Args:
            functions: Shape `(n_realizations,)` — per-realization values.
            weights:   Shape `(n_realizations,)` — realization weights.

        Returns:
            Aggregated value (scalar or 1-D array).
        """

    @abstractmethod
    def calculate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate gradients across realizations.

        When `merge_realizations` is `False` (default), `gradient` has shape
        `(n_realizations, n_variables)` and must be combined using `weights`.
        When `True`, a single pre-merged gradient of shape `(n_variables,)` is
        passed. Incompatible estimators should raise `ValueError` from `init`.

        Args:
            functions: Shape `(n_realizations,)` — needed for chain-rule
                estimators (e.g., standard deviation).
            gradient:  Shape `(n_realizations, n_variables)` or
                `(n_variables,)` if merged.
            weights:   Shape `(n_realizations,)` — realization weights.

        Returns:
            1-D array of shape `(n_variables,)`.
        """
