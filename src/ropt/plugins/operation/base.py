"""Base classes for operations and operation plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from ropt.plugins.base import Plugin
from ropt.plugins.event_handler import EventHandler


class OperationPlugin(Plugin):
    """Abstract base class for plugins that create Operation instances.

    This class defines the interface for plugins that act as factories for
    [`Operation`][ropt.plugins.operation.base.Operation] objects.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> Operation:
        """Create a Operation instance.

        This abstract class method serves as a factory for creating concrete
        [`Operation`][ropt.plugins.operation.base.Operation] objects. Plugin
        implementations must override this method to return an instance of
        their specific `Operation` subclass.

        The `name` argument specifies the requested operation, potentially in
        the format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created operation if the
        plugin supports multiple operation types.

        Args:
            name:   The requested operation name (potentially plugin-specific).
            kwargs: Additional arguments for custom configuration.

        Returns:
            An initialized instance of a `Operation` subclass.
        """


class Operation(ABC):
    """Abstract base class for optimization operations.

    This class defines the fundamental interface for all executable operations
    within an optimization workflow. Concrete implementations, which perform
    specific actions like running an optimizer or evaluating functions, must
    inherit from this base class.

    `Operation` objects are typically created by corresponding Subclasses must
    implement the abstract [`run`][ropt.plugins.operation.base.Operation.run]
    method to define specific behavior.
    """

    def __init__(self) -> None:
        """Initialize the Operation."""
        self._event_handlers: list[EventHandler] = []

    def add_event_handler(self, handler: EventHandler) -> Self:
        """Add a handler.

        Args:
            handler: The handler to add.
        """
        if isinstance(handler, EventHandler):
            self._event_handlers.append(handler)
        return self

    @property
    def event_handlers(self) -> list[EventHandler]:
        """Get the event handlers attached to this operation.

        Returns:
            A list of handlers.
        """
        return self._event_handlers

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Execute the logic defined by this operation.

        This abstract method must be implemented by concrete `Operation`
        subclasses to define the specific action the operation performs within
        the optimization workflow.

        The return value and type can vary depending on the specific
        implementation.

        Args:
            args:   Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The result of the execution, if any.
        """
