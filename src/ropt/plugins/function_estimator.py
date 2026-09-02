"""Plugin support for function estimators.

A function estimator combines the objective and gradient values of a set of
realizations into a single value. A
[`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.FunctionEstimatorPlugin]
is a factory that creates the
[`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] objects doing
the actual work, which the
[`PluginManager`][ropt.plugins.manager.PluginManager] discovers through the
`ropt.plugins.function_estimator` entry point group.

`ropt` ships
[`DefaultFunctionEstimator`][ropt.function_estimator.default.DefaultFunctionEstimator],
which calculates the weighted mean or standard deviation of the realizations.

See [Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

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
