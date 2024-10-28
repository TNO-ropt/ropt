"""This module defines the base classes for optimization plan objects.

Optimization plan objects can be added via the plugin mechanism to implement
additional functionality. This is done by creating a plugin class that derives
from the [`PlanPlugin`][ropt.plugins.plan.base.PlanPlugin] class. It
needs to define a [`create`][ropt.plugins.plan.base.PlanPlugin].create
method that generates the plan objects.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Union

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config.plan import ResultHandlerConfig, RunStepConfig
    from ropt.plan import OptimizerContext, ResultHandler, RunStep


class PlanPlugin(Plugin):
    """The abstract base class for plan plugins."""

    @abstractmethod
    def create(
        self,
        config: Union[ResultHandlerConfig, RunStepConfig],
        context: OptimizerContext,
    ) -> Union[ResultHandler, RunStep]:
        """Create the plan object.

        Args:
            config:  The configuration of the plan object
            context: The context in which the plan operates
        """
