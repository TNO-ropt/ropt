"""Base class for function estimator plugins."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config import FunctionEstimatorConfig
    from ropt.function_estimator import FunctionEstimator


class FunctionEstimatorPlugin(Plugin):
    """Abstract Base Class for Function Estimator Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator]
    instances. These plugins act as factories for specific function estimation
    strategies.

    During optimization execution, the
    [`PluginManager`][ropt.plugins.manager.PluginManager] identifies the
    appropriate function estimator plugin based on the configuration and uses
    its `create` class method to instantiate the actual `FunctionEstimator`
    object that will perform the aggregation of ensemble results (function
    values and gradients).
    """

    @classmethod
    @abstractmethod
    def create(cls, estimator_config: FunctionEstimatorConfig) -> FunctionEstimator:
        """Factory method to create a concrete FunctionEstimator instance.

        This abstract class method serves as a factory for creating concrete
        [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator]
        objects. Plugin implementations must override this method to return an
        instance of their specific `FunctionEstimator` subclass.

        The [`PluginManager`][ropt.plugins.manager.PluginManager] calls this
        method when the optimization requires a function estimator provided by
        this plugin.

        Args:
            estimator_config: The configuration object for this function estimator.

        Returns:
            An initialized FunctionEstimator object ready for use.
        """
