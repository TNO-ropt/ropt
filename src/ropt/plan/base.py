"""This module defines the base classes for optimization plan objects.

Optimization plan objects can be added via the plugin mechanism to implement
additional functionality. This is done by creating a plugin class that derives
from the [`PlanPlugin`][ropt.plugins.plan.base.PlanPlugin] class. It
needs to define a [`create`][ropt.plugins.plan.base.PlanPlugin].create
method that generates the plan objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ropt.config.plan import PlanStepConfig, ResultHandlerConfig
    from ropt.plan import Event, Plan


class ResultHandler(ABC):
    """Base class for results handler objects."""

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize the results handler object.

        The `config` and `plan` arguments are accessible as `handler_config`
        and `plan` properties.

        Args:
            config: The configuration of the handler object.
            plan:   The parent plan that contains the object.
        """
        self._handler_config = config
        self._plan = plan

    @property
    def handler_config(self) -> ResultHandlerConfig:
        """Return the handler object configuration.

        Returns:
            The configuration object.
        """
        return self._handler_config

    @property
    def plan(self) -> Plan:
        """Return the plan object.

        Returns:
            The plan object.
        """
        return self._plan

    @abstractmethod
    def handle_event(self, event: Event) -> Event:
        """Handle and propagate an event.

        Returns:
            The, possibly modified, event.
        """


class PlanStep(ABC):
    """Base class for plan steps."""

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize the plan object.

        Args:
            config: The configuration of the plan object.
            plan:   The parent plan.
        """
        self._step_config = config
        self._plan = plan

    @property
    def step_config(self) -> PlanStepConfig:
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
