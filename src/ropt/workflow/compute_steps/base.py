"""Base classes for compute steps and compute step plugins."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ropt.workflow.event_handlers import EventHandler

if TYPE_CHECKING:
    from collections.abc import Iterator


class ComputeStep(ABC):
    """Abstract base class for optimization compute steps.

    This class defines the fundamental interface for all executable compute steps
    within an optimization workflow. Concrete implementations, which perform
    specific actions like running an optimizer or evaluating functions, must
    inherit from this base class.

    A compute step instance may only have one active `run` at a time; its
    `run` method acquires a guard that raises a `RuntimeError` if the same
    instance is already running on another thread.
    """

    def __init__(self) -> None:
        """Initialize the ComputeStep."""
        self._event_handlers: list[EventHandler] = []
        self._running = False
        self._run_lock = threading.Lock()

    @contextmanager
    def _running_guard(self) -> Iterator[None]:
        with self._run_lock:
            if self._running:
                msg = "The compute step is already running on another thread."
                raise RuntimeError(msg)
            self._running = True
        try:
            yield
        finally:
            with self._run_lock:
                self._running = False

    def __getstate__(self) -> dict[str, Any]:  # ruff: ignore[undocumented-magic-method]
        state = self.__dict__.copy()
        state.pop("_run_lock", None)
        state.pop("_running", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:  # ruff: ignore[undocumented-magic-method]
        self.__dict__.update(state)
        self._running = False
        self._run_lock = threading.Lock()

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
            handler.register_compute_step()
            self._event_handlers.append(handler)

    @property
    def event_handlers(self) -> list[EventHandler]:
        """The event handlers attached to this compute step.

        Returns:
            A list of handlers.
        """
        return self._event_handlers

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
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
