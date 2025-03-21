"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Self

from ropt.enums import EventType
from ropt.plugins import PluginManager

if TYPE_CHECKING:
    from ropt.evaluator import Evaluator
    from ropt.plan import Event


class OptimizerContext:
    """Context class for shared state across a plan.

    An optimizer context object holds the information and state shared across
    all steps in an optimization plan. This currently includes the following:

    - An [`Evaluator`][ropt.evaluator.Evaluator] callable for evaluating
      functions.
    - A plugin manager to retrieve plugins used by the plan and optimizers.
    - Event callbacks that are triggered in response to specific events,
      executed after the plan has processed them.
    - An iterator producing unique evaluation ID's.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        """Initialize the optimization context.

        Initializes shared state and resources needed across an optimization
        plan, including a function evaluator and an optional expression
        evaluator for processing plan-specific expressions.

        Args:
            evaluator:      A callable used to evaluate functions within the plan.
            plugin_manager: Optional plugin manager.
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
        """Add an observer function.

        Observer functions are called during optimization when an event of the
        specified type occurs. The provided callable must accept a single
        argument of the [`Event`][ropt.plan.Event] class, which contains
        information about the occurred event.

        Note:
            Before the observer functions are called, all result handlers are
            executed, which may potentially modify the event.

        Args:
            event:    The type of events that the observer will react to.
            callback: The function to call when the specified event is received.
                      This function should accept one argument, which will be
                      an instance of the [`Event`][ropt.plan.Event] class.

        Returns:
            Self, to allow for method chaining.
        """
        self._subscribers[event].append(callback)
        return self

    def call_observers(self, event: Event) -> None:
        """Call observers for a specified event.

        This method invokes all observers associated with the context for the
        type of event passed as an argument.

        Args:
            event: The event that is emitted.
        """
        for callback in self._subscribers[event.event_type]:
            callback(event)
