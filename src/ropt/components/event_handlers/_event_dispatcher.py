"""This module implements the event dispatcher.

A dispatcher is what lets one handler serve runs on several threads: the events
are queued to its event loop and processed one at a time, so a handler never
sees two at once and needs no locking of its own.

The emitting run does not go on in the meantime. `dispatch_event` blocks it
until the event has been handled, which keeps the ordering a handler sees
meaningful and keeps a handler's exception on the stack of the run that caused
it, instead of tearing down the task group the dispatcher runs in.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Final

from ropt.components._loop import on_loop_thread, schedule
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
        # Ids of the pool threads, which run handlers and nothing else. A thread
        # only ever asks about itself, and registered its own id before it could
        # run anything, so it always sees at least that id without locking.
        self._handler_ids: frozenset[int] = frozenset()
        self._handler_ids_lock = threading.Lock()

    def add_event_handler(
        self, handler: EventHandler, *, run_in_thread: bool = False
    ) -> None:
        """Add an event handler.

        By default the handler is called directly on the event loop's thread.
        Pass `run_in_thread=True` for handlers that perform blocking I/O; see
        [Thread-based dispatch](../workflows/parallel.md#thread-based-dispatch).

        Args:
            handler:       The handler to add.
            run_in_thread: If True, dispatch via the dispatcher's thread pool
                           instead of the event loop.
        """
        handler._register_dispatcher()  # ruff: ignore[private-member-access]
        # Replaced rather than appended to: the loop thread iterates this list
        # while another thread may be adding to it, and a new list leaves any
        # iteration in progress on the one it started with.
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

        Events are handled in submission order. A handler exception is
        re-raised here, on the caller's own stack. See
        [Handler failures](../workflows/parallel.md#handler-failures).

        Args:
            event: The event to submit.

        Raises:
            WorkflowError: If the dispatcher is not running, or if the call is
                           made from the thread running its event loop, or from
                           one of its handler threads.
            Exception:    Whatever a handler raised while processing the event.
        """  # ruff: ignore[docstring-extraneous-exception]
        if not self._running.is_set():
            msg = "The event dispatcher is not running."
            raise WorkflowError(msg)
        if self._would_deadlock():
            msg = (
                "This dispatcher cannot be used from the thread running its "
                "loop, or from one of its handler threads: the call would wait "
                "for the loop that has to serve it."
            )
            raise WorkflowError(msg)
        assert self._loop is not None
        dispatch = self._dispatch(event)
        try:
            future = asyncio.run_coroutine_threadsafe(dispatch, self._loop)
        except RuntimeError as exc:
            # The loop went away between the check above and the handover.
            dispatch.close()
            msg = "The event dispatcher stopped."
            raise WorkflowError(msg) from exc
        # Blocks the emitting run until every handler is done with the event,
        # and re-raises whatever a handler raised.
        future.result()

    async def _dispatch(self, event: EnOptEvent) -> None:
        # A future per event, resolved once its handlers have finished: that is
        # what the caller in `dispatch_event` is waiting on.
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
        # A `None` through the queue, rather than a cancellation: it stops the
        # processing loop only after the events already queued are handled.
        if self._queue is not None:
            schedule(self._loop, self._queue.put_nowait, None)

    def _would_deadlock(self) -> bool:
        # The two places a handler runs, but stated as what it enforces: any
        # caller on the loop thread waits for the loop that must serve the
        # dispatch, whether it is a handler of this dispatcher or not.
        return on_loop_thread(self._loop) or threading.get_ident() in self._handler_ids

    def _handler_pool(self) -> ThreadPoolExecutor:
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=_MAX_HANDLER_THREADS,
                thread_name_prefix="ropt-handler",
                initializer=self._register_handler_thread,
            )
        return self._thread_pool

    def _register_handler_thread(self) -> None:
        with self._handler_ids_lock:
            self._handler_ids |= {threading.get_ident()}

    def _shutdown_pool(self) -> None:
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None
        # The ids belong to that pool's threads: a restarted dispatcher builds a
        # new pool, and the system may reuse the ids of the old one.
        with self._handler_ids_lock:
            self._handler_ids = frozenset()

    async def _run_handler(
        self, handler: EventHandler, event: EnOptEvent, *, run_in_thread: bool
    ) -> Exception | None:
        # Returns the exception instead of raising it, so one failing handler
        # does not keep the others from seeing the event.
        try:
            if run_in_thread:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._handler_pool(), partial(handler.handle_event, event)
                )
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
        # One event, one outcome for its caller: the first failure is the one
        # reported, the rest are only logged.
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
                    # Stop requested: mark it stopped so nothing new is queued,
                    # then still handle what is already in the queue.
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
            # However this ends, no caller may be left waiting on a future that
            # nothing will resolve any more.
            self._running.clear()
            self._reject_queued()
            self._shutdown_pool()
