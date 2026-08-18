"""Base classes for compute steps and compute step plugins."""

from __future__ import annotations

import functools
import threading
from abc import ABC, abstractmethod
from typing import Any

from ropt.components._transferred import _make_placeholder
from ropt.components.event_handlers import EventHandler
from ropt.exceptions import WorkflowError


class ComputeStep(ABC):
    """Abstract base class for optimization compute steps.

    This class defines the fundamental interface for all executable compute steps
    within an optimization workflow. Concrete implementations, which perform
    specific actions like running an optimizer or evaluating functions, must
    inherit from this base class.

    A compute step instance may only have one active `run` at a time. The guard
    is applied automatically to every subclass's `run` method, which raises a
    `WorkflowError` if the same instance is already running on another thread.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:  # ruff: ignore[undocumented-magic-method]
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__ and not getattr(
            cls.__dict__["run"], "__wrapped__", None
        ):
            original = cls.__dict__["run"]

            @functools.wraps(original)
            def _guarded(
                self: ComputeStep,
                *args: Any,  # ruff: ignore[any-type]
                _orig: Any = original,  # ruff: ignore[any-type]
                **kwargs: Any,  # ruff: ignore[any-type]
            ) -> Any:  # ruff: ignore[any-type]
                with self._run_lock:
                    if self._running:
                        msg = "The compute step is already running on another thread."
                        raise WorkflowError(msg)
                    self._running = True
                self._stop_flag.clear()
                try:
                    return _orig(self, *args, **kwargs)
                finally:
                    with self._run_lock:
                        self._running = False

            cls.run = _guarded  # type: ignore[method-assign]

    def __init__(self) -> None:
        """Initialize the ComputeStep."""
        self._event_handlers: list[EventHandler] = []
        self._running = False
        self._run_lock = threading.Lock()
        self._stop_flag = threading.Event()

    def __reduce__(self) -> tuple[object, tuple[str]]:  # ruff: ignore[undocumented-magic-method]
        return (_make_placeholder, ("A compute step",))

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add an event handler.

        Compute steps emit [`events`][ropt.events.EnOptEvent] to report on the
        calculations they perform. These events are processed by independently
        created [`event handlers`][ropt.components.event_handlers.EventHandler].
        Use the `add_event_handler` method to attach these handlers to the
        compute step.

        Args:
            handler: The handler to add.
        """
        if isinstance(handler, EventHandler):
            handler._register_compute_step()  # ruff: ignore[private-member-access]
            self._event_handlers.append(handler)

    @property
    def event_handlers(self) -> list[EventHandler]:
        """The event handlers attached to this compute step.

        Returns:
            A list of handlers.
        """
        return self._event_handlers

    def stop(self) -> None:
        """Request that this run stop gracefully at the next safe point.

        Intended for an event handler that decides, after inspecting an event,
        that its optimization should stop; the run then ends with
        `ExitCode.USER_ABORT`. Setting the request is thread-safe, so a handler
        running behind an event dispatcher may call it too. A new `run` clears
        any earlier request.
        """
        self._stop_flag.set()

    @property
    def stopped(self) -> bool:
        """Whether a stop has been requested for the current run.

        Returns:
            `True` if `stop` has been called since the run started.
        """
        return self._stop_flag.is_set()

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
