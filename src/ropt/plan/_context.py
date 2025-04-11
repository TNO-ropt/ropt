"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Self

from ropt.enums import EventType
from ropt.plugins import PluginManager

if TYPE_CHECKING:
    from ropt.evaluator import Evaluator
    from ropt.plan import Event


class OptimizerContext:
    """Manages shared state and resources for an optimization plan.

    The OptimizerContext acts as a central hub for managing shared resources and
    state across all steps within an optimization plan. This ensures that
    different parts of the plan can access and interact with the same
    information and tools.

    This context object is responsible for:

    - Providing a callable `Evaluator` for evaluating functions, which is
      essential for optimization algorithms to assess the quality of solutions.
      The evaluator is used to calculate the objective function's value and
      potentially constraint values for given variables.
    - Managing a `PluginManager` to retrieve and utilize plugins, allowing for
      extensibility and customization of the optimization workflow. Plugins are
      modular pieces of code that extend the functionality of the optimization
      framework, such as new `PlanStep` or `PlanHandler` implementations.
    - Handling event callbacks that are triggered in response to specific events
      during the plan's execution. These callbacks are executed after the plan
      has processed the event, allowing for actions to be taken in response to
      changes or milestones. This allows you to monitor the optimization, react
      to changes, or perform custom actions at specific points in the plan's
      execution.

    Args:
        evaluator:      A callable for evaluating functions in the plan.
        plugin_manager: A plugin manager; a default is created if not provided.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        """Initializes the optimization context.

        Sets up the shared state and resources required for an optimization
        plan. This includes a function evaluator and a plugin manager.

        Args:
            evaluator:      A callable for evaluating functions in the plan.
            plugin_manager: A plugin manager; a default is created if not provided.
        """
        self.evaluator = evaluator
        self.plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {
            event: [] for event in EventType
        }

    def add_observer(
        self,
        event: EventType,
        callback: Callable[[Event], None],
    ) -> Self:
        """Adds an observer function for a specific event type.

        Observer functions are called when an event of the specified type
        occurs during optimization. The provided callback function will receive
        an [`Event`][ropt.plan.Event] object containing information about the
        event.

        Args:
            event:    The type of event to observe.
            callback: The function to call when the event occurs.

        Returns:
            The OptimizerContext instance, allowing for method chaining.
        """
        self._subscribers[event].append(callback)
        return self

    def call_observers(self, event: Event) -> None:
        """Calls all observers for a specific event.

        This method triggers all observer functions registered for the given
        event type.

        Args:
            event: The event to emit to the observers.
        """
        for callback in self._subscribers[event.event_type]:
            callback(event)
