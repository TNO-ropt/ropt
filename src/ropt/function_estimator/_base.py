"""Abstract base class for function estimator implementations.

Function estimators are core components that determine how per-realization
objective/constraint function values and gradients are combined into the single
representative values used by the optimizer. This module provides the interface
that all concrete estimator implementations must follow.
"""

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

    All concrete function estimator implementations must inherit from this class
    and implement the required aggregation methods. Estimators are responsible
    for combining per-realization objective/constraint function values and
    gradients into single representative values using realization weights.

    The aggregated values produced by estimators are used directly by the
    optimizer during each iteration.

    **Lifecycle**

    1. Instantiation via `__init__`: Called by the plugin system with a
       configuration object.
    2. Setup via `init`: Called once per optimization workflow with the
       [`EnOptContext`][ropt.context.EnOptContext], allowing final initialization
       based on the full optimization configuration.
    3. Evaluation via `calculate_function` and `calculate_gradient`: Called
       repeatedly during optimization as new function/gradient values become
       available.

    Subclasses must implement:

    - `__init__`: Stores estimator configuration and performs lightweight setup.
    - `init`: Receives the optimization context for context-dependent initialization.
    - `calculate_function`: Combines function values from all realizations.
    - `calculate_gradient`: Combines gradients from all realizations.
    """

    @abstractmethod
    def __init__(self, estimator_config: FunctionEstimatorConfig) -> None:
        """Create a new function estimator instance.

        Called by the plugin system during instantiation. Subclasses should
        store the configuration and perform any lightweight initialization.
        Validation and context-dependent setup should be deferred to the
        `init` method.

        Args:
            estimator_config: Configuration object specifying the estimator
                method and any method-specific options.
        """

    @abstractmethod
    def init(self, context: EnOptContext) -> None:
        """Finalize initialization after the optimization context is known.

        Called once at the start of each optimization workflow, after all
        configuration is finalized. Use this method to perform context-dependent
        initialization such as validation of gradient settings, setup of
        internal state, or computation of method-specific parameters.

        Args:
            context: The full optimization context, containing all configuration
                and state for the current workflow.
        """

    @abstractmethod
    def calculate_function(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Aggregate function values across realizations into a single value.

        Combines per-realization objective or constraint function values into
        one representative value using the provided realization weights. The
        aggregation method is defined by the concrete estimator implementation.

        Args:
            functions: Array of shape `(n_realizations,)` containing function
                values from each realization.
            weights: Array of shape `(n_realizations,)` containing the weight
                for each realization. Typically represents the probability mass
                or importance of each realization.

        Returns:
            Aggregated function value as a scalar or 1D array.
        """

    @abstractmethod
    def calculate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate gradients across realizations into a single gradient.

        Combines per-realization gradients into one representative gradient
        vector using the provided realization weights and potentially the
        function values themselves. The aggregation method is defined by the
        concrete estimator implementation.

        This method is called after function values have been evaluated for all
        realizations. Some estimators (e.g., standard deviation) require the
        function values to correctly compute gradients via the chain rule.

        Args:
            functions: Array of shape `(n_realizations,)` containing function
                values from each realization. Used by some estimators to compute
                gradients correctly (e.g., chain rule for standard deviation).
            gradient: Array of shape `(n_realizations, n_variables)` containing
                gradient estimates for each realization. Depending on the
                [`GradientConfig.merge_realizations`][ropt.config.GradientConfig]
                setting, this may be either per-realization or pre-merged estimates
                (see note below).
            weights: Array of shape `(n_realizations,)` containing the weight
                for each realization.

        Returns:
            Aggregated gradient as a 1D array of shape `(n_variables,)`.

        Note: `merge_realizations` setting
            The [`GradientConfig.merge_realizations`][ropt.config.GradientConfig]
            flag determines how gradient inputs are prepared:

            - If `False` (default): `ropt` estimates a separate gradient for each
              realization with non-zero weight. The implementation must combine
              these per-realization gradients using `weights` (e.g., weighted
              average or chain-rule-adjusted aggregation).
            - If `True`: `ropt` estimates a single merged gradient by treating
              all perturbations across all realizations collectively, then passes
              that single estimate. The implementation should handle this
              appropriately (e.g., return it unchanged for averaging-like
              operations).

            The `merge_realizations=True` option is useful for workflows with
            few perturbations (even just one) but is only suitable for estimators
            performing simple averaging. Implementations should validate
            compatibility during `init` and raise `ValueError` if the configured
            setting is incompatible with the estimator's logic (e.g., standard
            deviation).
        """
