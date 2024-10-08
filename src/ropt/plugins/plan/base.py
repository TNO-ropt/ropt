"""This module defines the base classes for optimization plan objects.

Optimization plan objects can be added via the plugin mechanism to implement
additional functionality. This is done by creating a plugin class that derives
from the [`PlanPlugin`][ropt.plugins.plan.base.PlanPlugin] class. It
needs to define a [`create`][ropt.plugins.plan.base.PlanPlugin].create
method that generates the plan objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config.plan import ContextConfig, StepConfig
    from ropt.plan import ContextUpdate, OptimizerContext, Plan


class ContextObj:
    """Base class for context objects."""

    def __init__(self, config: ContextConfig, plan: Plan) -> None:
        """Initialize the context object.

        The `config` and `plan` arguments are accessible as `context_config`
        and `plan` properties.

        Args:
            config: The configuration of the context object
            plan:   The parent plan that contains the object
        """
        self._context_config = config
        self._plan = plan

    def update(self, value: ContextUpdate) -> None:
        """Update the value of the object.

        Args:
            value: The value used for the update.
        """

    def reset(self) -> None:
        """Resets the object to its initial state."""

    @property
    def context_config(self) -> ContextConfig:
        """Return the context object configuration.

        Returns:
            The configuration object.
        """
        return self._context_config

    @property
    def plan(self) -> Plan:
        """Return the plan object.

        Returns:
            The plan object.
        """
        return self._plan

    def get_variable(self) -> Any:  # noqa: ANN401
        """Get a variable with the name equal to the context object ID.

        Returns:
            The value of the variable.
        """
        return self._plan[self._context_config.id]

    def set_variable(self, value: Any) -> None:  # noqa: ANN401
        """Set a variable with the name equal to the context object ID.

        Args:
            value: The value
        """
        self._plan[self._context_config.id] = value


class PlanStep(ABC):
    """Base class for plan steps."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize the plan object.

        Args:
            config: The configuration of the plan object
            plan:   The parent plan
        """
        self._step_config = config
        self._plan = plan

    @property
    def step_config(self) -> StepConfig:
        """Return the step object config.

        Returns:
            The configuration object.
        """
        return self._step_config

    @property
    def plan(self) -> Plan:
        """Return the plan object.

        Returns:
            The plan object.
        """
        return self._plan

    @abstractmethod
    def run(self) -> None:
        """Run the step object."""


class PlanPlugin(Plugin):
    """The abstract base class for plan plugins."""

    @abstractmethod
    def create(
        self, config: Union[ContextConfig, StepConfig], context: OptimizerContext
    ) -> Union[ContextObj, PlanStep]:
        """Create the plan object.

        Args:
            config:  The configuration of the plan object
            context: The context in which the plan operates
        """
