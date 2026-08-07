"""This module implements the event dispatcher."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING

from ropt.components._transferred import _make_placeholder

if TYPE_CHECKING:
    from ropt.events import EnOptEvent

    from .base import EventHandler

    _QueueItem = tuple[EnOptEvent, Future[None]]

_logger = logging.getLogger(__name__)


class EventDispatcher:
    """Dispatches events to handlers from the asyncio event loop's thread.

    See [Parallel Evaluation](../usage/parallel.md#event-dispatcher) for usage.
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[EventHandler, bool]] = []
        self._queue: asyncio.Queue[_QueueItem | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = threading.Event()

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

    def remove_event_handler(self, handler: EventHandler) -> None:
        """Remove a previously added handler.

        The handler is released, so it can afterwards be added to another
        dispatcher or registered with a compute step.

        Args:
            handler: The handler to remove.

        Raises:
            ValueError: If the handler was not added to this dispatcher.
        """
        remaining = [item for item in self._handlers if item[0] is not handler]
        if len(remaining) == len(self._handlers):
            msg = "This handler was not added to the dispatcher."
            raise ValueError(msg)
        self._handlers = remaining
        handler.unregister_dispatcher()

    def dispatch_event(self, event: EnOptEvent) -> None:
        """Submit an event and block until every handler has processed it.

        The event is queued to the dispatcher and this call blocks on the
        calling thread until the dispatcher has finished handling it. Events are
        handled in submission order. If a handler raises, the original exception
        is re-raised here — on the caller's own stack — so it surfaces as a
        clean, single exception, mirroring how the executor's
        [`put_error`][ropt.components.executors.Task.put_error] is re-raised by the
        awaiting evaluator.

        Args:
            event: The event to submit.

        Raises:
            RuntimeError: If the dispatcher is not running.
            Exception:    Whatever a handler raised while processing the event.
        """  # ruff: ignore[docstring-extraneous-exception]
        if not self._running.is_set():
            msg = "Cannot submit an event to an EventDispatcher that is not running."
            raise RuntimeError(msg)
        assert self._loop is not None
        assert self._queue is not None
        future: Future[None] = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (event, future))
        future.result()

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

    @staticmethod
    async def _run_handler(
        handler: EventHandler, event: EnOptEvent, *, run_in_thread: bool
    ) -> Exception | None:
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
            return exc
        return None

    async def _run_handlers_and_complete_future(
        self, event: EnOptEvent, future: Future[None]
    ) -> None:
        results = await asyncio.gather(
            *(
                self._run_handler(handler, event, run_in_thread=run_in_thread)
                for handler, run_in_thread in self._handlers
                if event.event_type in handler.event_types
            )
        )
        if future.done():
            return
        error = next((result for result in results if result is not None), None)
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)

    async def _drain(self) -> None:
        assert self._queue is not None
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self._queue.task_done()
            if item is not None:
                await self._run_handlers_and_complete_future(item[0], item[1])

    @staticmethod
    def _reject(future: Future[None]) -> None:
        if not future.done():
            future.set_exception(RuntimeError("The event dispatcher stopped."))

    def _reject_queued(self) -> None:
        assert self._queue is not None
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self._queue.task_done()
            if item is not None:
                self._reject(item[1])

    async def _process(self) -> None:
        assert self._queue is not None
        pending: Future[None] | None = None
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            while True:
                item = await self._queue.get()
                self._queue.task_done()
                if item is None:
                    await self._drain()
                    break
                pending = item[1]
                await self._run_handlers_and_complete_future(item[0], item[1])
                pending = None
        except BaseException:
            if pending is not None:
                self._reject(pending)
            self._reject_queued()
            raise
        finally:
            self._running.clear()
