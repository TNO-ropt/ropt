"""This module implements the event dispatcher."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Final

from ropt.components._loop import schedule
from ropt.components._transferred import _make_placeholder
from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from ropt.events import EnOptEvent

    from .base import EventHandler

    _QueueItem = tuple[EnOptEvent, asyncio.Future[None]]

_logger = logging.getLogger(__name__)

# Deliberately generous: handlers can be added after the pool exists and pools
# cannot be resized, while a pool smaller than the handlers matching one event
# deadlocks if those handlers wait on each other. Threads are created on demand,
# so a ceiling that is never reached costs nothing.
_MAX_HANDLER_THREADS: Final = 256


class EventDispatcher:
    """Dispatches events to handlers from the asyncio event loop's thread.

    Handlers added with `run_in_thread=True` run on a thread pool the
    dispatcher owns and shuts down when it stops, so handler work is isolated
    from the asyncio loop's shared default pool.

    See [Parallel Evaluation](../workflows/parallel.md#event-dispatcher) for usage.
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[EventHandler, bool]] = []
        self._queue: asyncio.Queue[_QueueItem | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = threading.Event()
        self._thread_pool: ThreadPoolExecutor | None = None
        # Marks the threads that are currently running a handler, so a nested
        # dispatch can be told apart from an ordinary one.
        self._local = threading.local()

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_make_placeholder, ("An event dispatcher",))

    def add_event_handler(
        self, handler: EventHandler, *, run_in_thread: bool = False
    ) -> None:
        """Add an event handler.

        By default the handler is called directly in the event loop's thread,
        which is efficient for handlers that only do in-memory work. Pass
        `run_in_thread=True` for handlers that perform blocking operations such
        as file I/O, database writes, or network calls. Such handlers run on a
        thread pool the dispatcher owns, so they never compete for the asyncio
        loop's shared default pool. Multiple handlers with `run_in_thread=True`
        that match the same event are dispatched in parallel via
        `asyncio.gather`.

        Args:
            handler:       The handler to add.
            run_in_thread: If True, dispatch via the dispatcher's thread pool
                           instead of the event loop.
        """
        handler._register_dispatcher()  # ruff: ignore[private-member-access]
        self._handlers = [*self._handlers, (handler, run_in_thread)]

    def remove_event_handler(self, handler: EventHandler) -> bool:
        """Remove a previously added handler.

        The handler is released, so it can afterwards be added to another
        dispatcher or registered with a compute step.

        Args:
            handler: The handler to remove.

        Returns:
            Whether the removed handler was set to run in a thread.

        Raises:
            WorkflowError: If the handler was not added to this dispatcher.
        """
        removed = [item for item in self._handlers if item[0] is handler]
        if not removed:
            msg = "This handler was not added to the dispatcher."
            raise WorkflowError(msg)
        self._handlers = [item for item in self._handlers if item[0] is not handler]
        handler._unregister_dispatcher()  # ruff: ignore[private-member-access]
        return removed[0][1]

    def dispatch_event(self, event: EnOptEvent) -> None:
        """Submit an event and block until every handler has processed it.

        The event is queued to the dispatcher and this call blocks on the
        calling thread until the dispatcher has finished handling it. Events are
        handled in submission order. If a handler raises, the original exception
        is re-raised here — on the caller's own stack — so it surfaces as a
        clean, single exception, mirroring how the executor's
        [`fail`][ropt.components.executors.Submission.fail] is re-raised by the
        awaiting evaluator.

        Args:
            event: The event to submit.

        Raises:
            WorkflowError: If the dispatcher is not running, or if a handler of
                           this dispatcher is dispatching.
            Exception:    Whatever a handler raised while processing the event.
        """  # ruff: ignore[docstring-extraneous-exception]
        if not self._running.is_set():
            msg = "The event dispatcher is not running."
            raise WorkflowError(msg)
        if getattr(self._local, "in_handler", False):
            msg = "A handler cannot dispatch on its own dispatcher."
            raise WorkflowError(msg)
        assert self._loop is not None
        dispatch = self._dispatch(event)
        try:
            future = asyncio.run_coroutine_threadsafe(dispatch, self._loop)
        except RuntimeError as exc:
            dispatch.close()
            msg = "The event dispatcher stopped."
            raise WorkflowError(msg) from exc
        future.result()

    async def _dispatch(self, event: EnOptEvent) -> None:
        if not self._running.is_set() or self._queue is None:
            msg = "The event dispatcher stopped."
            raise WorkflowError(msg)
        handled = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((event, handled))
        await handled

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the dispatcher.

        Args:
            task_group: The task group to use.

        Raises:
            WorkflowError: If the dispatcher is already running.
        """
        if self._running.is_set():
            msg = "The event dispatcher is already running."
            raise WorkflowError(msg)
        self._queue = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._running.set()
        task_group.create_task(self._process())

    def cancel(self) -> None:
        """Stop the dispatcher.

        May be called from any thread.
        """
        if self._queue is not None:
            schedule(self._loop, self._queue.put_nowait, None)

    def _handler_pool(self) -> ThreadPoolExecutor:
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=_MAX_HANDLER_THREADS, thread_name_prefix="ropt-handler"
            )
        return self._thread_pool

    def _shutdown_pool(self) -> None:
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None

    def _invoke(self, handler: EventHandler, event: EnOptEvent) -> None:
        self._local.in_handler = True
        try:
            handler.handle_event(event)
        finally:
            self._local.in_handler = False

    async def _run_handler(
        self, handler: EventHandler, event: EnOptEvent, *, run_in_thread: bool
    ) -> Exception | None:
        try:
            if run_in_thread:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._handler_pool(), partial(self._invoke, handler, event)
                )
            else:
                self._invoke(handler, event)
        except Exception as exc:
            _logger.exception(
                "Event handler %r failed while handling %s",
                handler,
                event.event_type,
            )
            return exc
        return None

    async def _run_handlers_and_complete_future(
        self, event: EnOptEvent, future: asyncio.Future[None]
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
    def _reject(future: asyncio.Future[None]) -> None:
        if not future.done():
            future.set_exception(WorkflowError("The event dispatcher stopped."))

    def _reject_queued(self) -> None:
        assert self._queue is not None
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self._queue.task_done()
            if item is not None:
                self._reject(item[1])

    async def _process(self) -> None:
        assert self._queue is not None
        pending: asyncio.Future[None] | None = None
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            while True:
                item = await self._queue.get()
                self._queue.task_done()
                if item is None:
                    self._running.clear()
                    await self._drain()
                    break
                pending = item[1]
                await self._run_handlers_and_complete_future(item[0], item[1])
                pending = None
        except asyncio.CancelledError:
            if pending is not None:
                self._reject(pending)
            raise
        except BaseException as exc:
            # A handler raising a BaseException is still fatal, but its caller
            # must see that error rather than a generic "stopped".
            if pending is not None and not pending.done():
                pending.set_exception(exc)
            raise
        finally:
            self._running.clear()
            self._reject_queued()
            self._shutdown_pool()
