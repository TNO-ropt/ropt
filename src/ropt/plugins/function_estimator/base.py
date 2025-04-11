"""This module defines the abstract base classes for function estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class FunctionEstimator(ABC):
    """Abstract Base Class for Function Estimator Implementations.

    This class defines the fundamental interface for all concrete function
    estimator implementations within the `ropt` framework. Function estimator
    plugins provide classes derived from `FunctionEstimator` that encapsulate
    the logic for combining the objective function values and gradients from an
    ensemble of realizations into a single representative value. This aggregated
    value is then used by the core optimization algorithm.

    Instances of `FunctionEstimator` subclasses are created by their
    corresponding
    [`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.base.FunctionEstimatorPlugin]
    factories. They are initialized with an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object detailing the
    optimization setup and the `estimator_index` identifying the specific
    estimator configuration to use from the config.

    The core functionality involves combining results using realization weights,
    performed by the `calculate_function` and `calculate_gradient` methods,
    which must be implemented by subclasses.

    Subclasses must implement:

    - `__init__`: To accept the configuration and index.
    - `calculate_function`: To combine function values from realizations.
    - `calculate_gradient`: To combine gradient values from realizations.

    """

    def __init__(self, enopt_config: EnOptConfig, estimator_index: int) -> None:  # noqa: B027
        """Initialize the function estimator object.

        The `function_estimators` field in the `enopt_config` is a tuple of
        estimator configurations
        ([`FunctionEstimatorConfig`][ropt.config.enopt.FunctionEstimatorConfig]).
        The `estimator_index` identifies which configuration from this tuple
        should be used to initialize this specific estimator instance.

        Args:
            enopt_config:    The configuration of the optimizer.
            estimator_index: The index of the estimator configuration to use.
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
            [`GradientConfig`][ropt.config.enopt.GradientConfig] determines how
            the initial gradient estimate(s) are computed by `ropt` *before*
            being passed to this `calculate_gradient` method.

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
            initialization (`__init__`) and raise a
            [`ConfigError`][ropt.exceptions.ConfigError] if `merge_realizations=True`
            is incompatible with the estimator's logic (e.g., standard deviation).

        Args:
            functions: The functions for each realization.
            gradient:  The gradient for each realization.
            weights:   The weight of each realization.

        Returns:
            The expected gradients.
        """


class FunctionEstimatorPlugin(Plugin):
    """Abstract Base Class for Function Estimator Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`FunctionEstimator`][ropt.plugins.function_estimator.base.FunctionEstimator]
    instances. These plugins act as factories for specific function estimation
    strategies.

    During plan execution, the [`PluginManager`][ropt.plugins.PluginManager]
    identifies the appropriate function estimator plugin based on the
    configuration and uses its `create` class method to instantiate the actual
    `FunctionEstimator` object that will perform the aggregation of ensemble
    results (function values and gradients).
    """

    @classmethod
    @abstractmethod
    def create(
        cls, enopt_config: EnOptConfig, estimator_index: int
    ) -> FunctionEstimator:
        """Factory method to create a concrete FunctionEstimator instance.

        This abstract class method serves as a factory for creating concrete
        [`FunctionEstimator`][ropt.plugins.function_estimator.base.FunctionEstimator]
        objects. Plugin implementations must override this method to return an
        instance of their specific `FunctionEstimator` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        an optimization step requires a function estimator provided by this
        plugin.

        Args:
            enopt_config:    The main EnOpt configuration object.
            estimator_index: Index into `enopt_config.function_estimators` for
                             this estimator.

        Returns:
            An initialized FunctionEstimator object ready for use.
        """
