"""Base classes for compute steps and compute step plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ropt.workflow.event_handlers import EventHandler


class ComputeStep(ABC):
    """Abstract base class for optimization compute steps.

    This class defines the fundamental interface for all executable compute steps
    within an optimization workflow. Concrete implementations, which perform
    specific actions like running an optimizer or evaluating functions, must
    inherit from this base class.
    """

    def __init__(self) -> None:
        """Initialize the ComputeStep."""
        self._event_handlers: list[EventHandler] = []

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add an event handler.

        Compute steps emit [`events`][ropt.events.EnOptEvent] to report on the
        calculations they perform. These events are processed by independently
        created [`event handlers`][ropt.workflow.event_handlers.EventHandler].
        Use the `add_event_handler` method to attach these handlers to the
        compute step.

        Args:
            handler: The handler to add.
        """
        if isinstance(handler, EventHandler):
            self._event_handlers.append(handler)

    @property
    def event_handlers(self) -> list[EventHandler]:
        """Get the event handlers attached to this compute step.

        Returns:
            A list of handlers.
        """
        return self._event_handlers

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Execute the logic defined by this compute step.

        This abstract method must be implemented by concrete `ComputeStep`
        subclasses to define the specific action the compute step performs within
        the optimization workflow.

        The return value and type can vary depending on the specific
        implementation.

        Args:
            args:   Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The result of the execution, if any.
        """
