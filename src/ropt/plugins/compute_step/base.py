"""Base classes for compute steps and compute step plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from ropt.plugins.base import Plugin
from ropt.plugins.event_handler import EventHandler


class ComputeStepPlugin(Plugin):
    """Abstract base class for plugins that create ComputeStep instances.

    This class defines the interface for plugins that act as factories for
    [`ComputeStep`][ropt.plugins.compute_step.base.ComputeStep] objects.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> ComputeStep:
        """Create a ComputeStep instance.

        This abstract class method serves as a factory for creating concrete
        [`ComputeStep`][ropt.plugins.compute_step.base.ComputeStep] objects. Plugin
        implementations must override this method to return an instance of
        their specific `ComputeStep` subclass.

        The `name` argument specifies the requested compute step, potentially in
        the format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created compute step if the
        plugin supports multiple compute step types.

        Args:
            name:   The requested compute step name (potentially plugin-specific).
            kwargs: Additional arguments for custom configuration.

        Returns:
            An initialized instance of a `ComputeStep` subclass.
        """


class ComputeStep(ABC):
    """Abstract base class for optimization compute steps.

    This class defines the fundamental interface for all executable compute steps
    within an optimization workflow. Concrete implementations, which perform
    specific actions like running an optimizer or evaluating functions, must
    inherit from this base class.

    `ComputeStep` instances are typically created using the
    [`create_compute_step`][ropt.plugins.PluginManager.create_compute_step] method
    of a plugin manager.
    """

    def __init__(self) -> None:
        """Initialize the ComputeStep."""
        self._event_handlers: list[EventHandler] = []

    def add_event_handler(self, handler: EventHandler) -> Self:
        """Add an event handler.

        Compute steps emit [`events`][ropt.workflow.Event] to report on the
        calculations they perform. These events are processed by independently
        created [`event
        handlers`][ropt.plugins.event_handler.base.EventHandler]. Use the
        `add_event_handler` method to attach these handlers to the compute step.

        Args:
            handler: The handler to add.
        """
        if isinstance(handler, EventHandler):
            self._event_handlers.append(handler)
        return self

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
