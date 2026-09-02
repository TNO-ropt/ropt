"""Base classes for compute steps and compute step plugins."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ropt.components.event_handlers import EventHandler
from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.context import EnOptContext
    from ropt.events import EnOptEvent

_ResultT = TypeVar("_ResultT")


class ComputeStep(ABC, Generic[_ResultT]):
    """Abstract base class for optimization compute steps.

    A concrete step performs a specific action, such as running an optimizer or
    evaluating functions, and implements it in `_run`. The public `run` adds the
    guard that a step instance may only have one active run at a time, raising a
    `WorkflowError` if the same instance is already running on another thread.

    The type parameter is what `run` returns.
    """

    def __init__(self) -> None:
        """Initialize the ComputeStep."""
        self._event_handlers: list[EventHandler] = []
        self._running = False
        self._run_lock = threading.Lock()
        self._stop_flag = threading.Event()

    def add_event_handler(self, handler: EventHandler) -> None:
        """Attach an event handler to receive this step's events.

        Args:
            handler: The handler to add.

        Raises:
            TypeError: If `handler` is not an event handler.
        """
        if not isinstance(handler, EventHandler):
            msg = f"Not an event handler: {type(handler).__name__}"
            raise TypeError(msg)
        handler._register_compute_step()  # ruff: ignore[private-member-access]
        self._event_handlers.append(handler)

    @property
    def event_handlers(self) -> list[EventHandler]:
        """The event handlers attached to this compute step.

        Returns:
            A list of handlers.
        """
        return self._event_handlers

    def _emit_event(self, event: EnOptEvent) -> None:
        # Handlers run inline, on this run's own stack: a local handler that
        # raises unwinds this run, and one behind a dispatcher blocks here until
        # the dispatcher has finished with the event.
        event.source = self
        for handler in self.event_handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)

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
    def _run(
        self,
        context: EnOptContext,
        variables: ArrayLike,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> _ResultT:
        """Execute the logic defined by this compute step.

        Implemented by concrete subclasses; callers use `run`, which adds the
        concurrency guard.

        Args:
            context:   The optimization context.
            variables: The initial variable values.
            metadata:  Optional metadata to attach to the results.

        Returns:
            The result of the execution, if any.
        """

    def run(
        self,
        context: EnOptContext,
        variables: ArrayLike,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> _ResultT:
        """Run this compute step.

        Args:
            context:   The optimization context.
            variables: The initial variable values.
            metadata:  Optional metadata to attach to the results.

        Returns:
            The result of the execution, if any.

        Raises:
            WorkflowError: If this instance is already running on another thread.
        """
        with self._run_lock:
            if self._running:
                msg = "The compute step is already running on another thread."
                raise WorkflowError(msg)
            self._running = True
        # A step reused after a stopped run must not start out stopped.
        self._stop_flag.clear()
        try:
            return self._run(context, variables, metadata=metadata)
        finally:
            with self._run_lock:
                self._running = False
