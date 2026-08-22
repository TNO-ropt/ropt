"""Base class for function estimator plugins."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config import FunctionEstimatorConfig
    from ropt.function_estimator import FunctionEstimator


class FunctionEstimatorPlugin(Plugin):
    """Abstract base class for function estimator plugins (factories).

    Creates [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator]
    instances; concrete plugins implement `create` as a factory for their own
    `FunctionEstimator` subclass.
    """

    @classmethod
    @abstractmethod
    def create(cls, estimator_config: FunctionEstimatorConfig) -> FunctionEstimator:
        """Create a FunctionEstimator instance.

        Called by the [`PluginManager`][ropt.plugins.manager.PluginManager]
        when the optimization requires a function estimator from this plugin.

        Args:
            estimator_config: The configuration object for this function estimator.

        Returns:
            An initialized FunctionEstimator object ready for use.
        """
