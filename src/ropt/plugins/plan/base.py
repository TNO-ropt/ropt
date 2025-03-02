"""This module defines the base classes for optimization plan plugins.

Optimization plan steps and result handlers can be added via the plugin
mechanism to implement additional functionality. This is done by creating plugin
classes that derive from the
[`PlanHandlerPlugin`][ropt.plugins.plan.base.PlanHandlerPlugin] and
[`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin] classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.plan import Event, Plan


class PlanHandlerPlugin(Plugin):
    """Abstract base class for plan handler plugins.

    This base class serves as the foundation for handler plugins used in an
    optimization plan. Any plugin derived from this class can be built-in,
    installed via a plugin mechanism, or dynamically loaded. During the
    execution of an optimization plan, the required plugin is located through
    the [`PluginManager`][ropt.plugins.PluginManager], which then uses the
    plugin's `create_*` functions to instantiate  a
    [`ResultHandler`][ropt.plugins.plan.base.ResultHandler].
    """

    @abstractmethod
    def create(
        self,
        name: str,
        plan: Plan,
        **kwargs: Any,  # noqa: ANN401
    ) -> ResultHandler:
        """Create a result handler.

        This factory function instantiates result handler
        object based on the provided configuration.

        Args:
            name:   The name of the handler.
            plan:   The plan in which the handler operates.
            kwargs: Additional arguments to pass to the handler.
        """


class PlanStepPlugin(Plugin):
    """Abstract base class for plan step plugins.

    This base class serves as the foundation for step plugins used in an
    optimization plan. Any plugin derived from this class can be built-in,
    installed via a plugin mechanism, or dynamically loaded. During the
    execution of an optimization plan, the required plugin is located through
    the [`PluginManager`][ropt.plugins.PluginManager], which then uses the
    plugin's `create_*` functions to instantiate a
    [`PlanStep`][ropt.plugins.plan.base.PlanStep].
    """

    @abstractmethod
    def create(
        self,
        name: str,
        plan: Plan,
        **kwargs: Any,  # noqa: ANN401
    ) -> PlanStep:
        """Create a result handler.

        This factory function instantiates result handler
        object based on the provided configuration.

        Args:
            name: The name of the step.
            plan: The plan in which the step operates.
            kwargs: Additional arguments to pass to the step.
        """


class PlanStep(ABC):
    """Base class for steps.

    steps loaded and created by the plugin manager must inherit from this
    abstract base class. It defines a `run` method that must be overridden to
    implement the specific functionality of each step.
    """

    def __init__(self, plan: Plan) -> None:
        """Initialize the step.

        The `plan` argument is accessible as the `plan` property.

        Args:
            plan:   The parent plan that manages this step.
        """
        self.__stored_plan = plan
        self.__stored_values: dict[str, Any] = {}

    @property
    def plan(self) -> Plan:
        """Return the plan that executes the step.

        Returns:
            The plan object.
        """
        return self.__stored_plan

    @abstractmethod
    def run(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Execute the step object.

        This method must be overloaded to implement the functionality of the
        step.

        Args:
            kwargs: Optional keyword arguments.
        """

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get the value of a plan variable.

        This method implements the `[]` operator on the step object to retrieve
        the value associated with a specific key.

        Args:
            key: The key to retrieve.

        Returns:
            The value corresponding to the key.
        """
        if key in self.__stored_values:
            return self.__stored_values[key]
        msg = f"Unknown plan variable: `{key}`"
        raise AttributeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set a plan variable to the given value.

        This method implements the `[]` operator on the step object to store
        arbitrary values.

        Args:
            key:   The key to set.
            value: The value to store.
        """
        if not key.isidentifier():
            msg = f"Not a valid variable name: `{key}`"
            raise AttributeError(msg)
        self.__stored_values[key] = value


class ResultHandler(ABC):
    """Base class for result handler objects.

    Result handlers loaded and created by the plugin manager must inherit from
    this abstract base class. It defines a `handle_event` method that must be
    overridden to implement specific functionality.
    """

    def __init__(self, plan: Plan) -> None:
        """Initialize a results handler.

        The `plan` argument is accessible as the `plan` property.

        Args:
            plan:   The parent plan that contains the object.
        """
        self.__stored_plan = plan
        self.__stored_values: dict[str, Any] = {}

    @property
    def plan(self) -> Plan:
        """Return the associated plan that executes the handler.

        Returns:
            The plan object.
        """
        return self.__stored_plan

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Handle and propagate an event.

        This method must be overloaded to implement the functionality of the
        handler, based on the information passed via the event object.
        """

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get the value of a plan variable.

        This method implements the `[]` operator on the handler object to retrieve
        the value associated with a specific key.

        Args:
            key: The key to retrieve.

        Returns:
            The value corresponding to the key.
        """
        if key in self.__stored_values:
            return self.__stored_values[key]
        msg = f"Unknown plan variable: `{key}`"
        raise AttributeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set a plan variable to the given value.

        This method implements the `[]` operator on the handler object to store
        arbitrary values.

        Args:
            key:   The key to set.
            value: The value to store.
        """
        if not key.isidentifier():
            msg = f"Not a valid variable name: `{key}`"
            raise AttributeError(msg)
        self.__stored_values[key] = value
