"""This module implements the event dispatcher."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING

from ropt.workflow._transferred import _make_placeholder

if TYPE_CHECKING:
    from ropt.events import EnOptEvent

    from .base import EventHandler

_logger = logging.getLogger(__name__)


class EventDispatcher:
    """Dispatches events to handlers from the asyncio event loop's thread.

    See [Parallel Evaluation](../usage/parallel.md#event-dispatcher) for usage.
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[EventHandler, bool]] = []
        self._queue: asyncio.Queue[EnOptEvent | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = threading.Event()
        self._errors: queue.Queue[Exception] = queue.Queue()

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_make_placeholder, ("An event dispatcher",))

    def add_event_handler(
        self, handler: EventHandler, *, run_in_thread: bool = False
    ) -> None:
        """Add an event handler.

        By default the handler is called directly in the event loop's thread,
        which is efficient for handlers that only do in-memory work. Pass
        `run_in_thread=True` for handlers that perform blocking operations such
        as file I/O, database writes, or network calls. Multiple handlers with
        `run_in_thread=True` that match the same event are dispatched in
        parallel via `asyncio.gather`.

        Args:
            handler:       The handler to add.
            run_in_thread: If True, dispatch via the thread pool instead of
                           the event loop.
        """
        handler.register_dispatcher()
        self._handlers.append((handler, run_in_thread))

    def put_event(self, event: EnOptEvent) -> None:
        """Submit an event from any thread.

        Args:
            event: The event to submit.

        Raises:
            RuntimeError: If the dispatcher is not running.
        """
        if not self._running.is_set():
            msg = "Cannot submit an event to an EventDispatcher that is not running."
            raise RuntimeError(msg)
        assert self._loop is not None
        assert self._queue is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def raise_pending_error(self) -> None:
        """Re-raise a recorded handler failure on the caller's stack.

        A failing event handler cannot raise from the dispatcher task without
        tearing down the session task group as a `BaseExceptionGroup`. Instead
        the failure is recorded, and the emitting run drains it here — on its
        own call stack — where the original exception propagates as a clean,
        single exception with a normal exit code, mirroring how the executor's
        [`put_error`][ropt.workflow.executors.Task.put_error] is re-raised by
        the awaiting evaluator. If no failure is pending this is a no-op.

        Raises:
            Exception: The original exception raised by a failing event handler.
        """  # ruff: ignore[docstring-extraneous-exception]
        try:
            exc = self._errors.get_nowait()
        except queue.Empty:
            return
        raise exc

    def is_running(self) -> bool:
        """Check if the dispatcher is running.

        Returns:
            True if the dispatcher is running.
        """
        return self._running.is_set()

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the dispatcher.

        Args:
            task_group: The task group to use.

        Raises:
            RuntimeError: If the dispatcher is already running.
        """
        if self._running.is_set():
            msg = "EventDispatcher is already running."
            raise RuntimeError(msg)
        self._queue = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._running.set()
        task_group.create_task(self._process())

    def cancel(self) -> None:
        """Stop the dispatcher."""
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    async def _dispatch(self, event: EnOptEvent) -> None:
        await asyncio.gather(
            *(
                self._run_handler(handler, event, run_in_thread=run_in_thread)
                for handler, run_in_thread in self._handlers
                if event.event_type in handler.event_types
            )
        )

    async def _run_handler(
        self, handler: EventHandler, event: EnOptEvent, *, run_in_thread: bool
    ) -> None:
        try:
            if run_in_thread:
                await asyncio.to_thread(handler.handle_event, event)
            else:
                handler.handle_event(event)
        except Exception as exc:
            _logger.exception(
                "Event handler %r failed while handling %s",
                handler,
                event.event_type,
            )
            self._errors.put(exc)

    async def _drain(self) -> None:
        assert self._queue is not None
        while not self._queue.empty():
            remaining = self._queue.get_nowait()
            self._queue.task_done()
            if remaining is not None:
                await self._dispatch(remaining)

    async def _process(self) -> None:
        assert self._queue is not None
        try:
            while True:
                event = await self._queue.get()
                self._queue.task_done()
                if event is None:
                    await self._drain()
                    break
                await self._dispatch(event)
        finally:
            self._running.clear()
