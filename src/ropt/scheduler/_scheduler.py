"""Scheduler class."""

import asyncio
import contextlib
import uuid
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Self

from ._events import ErrorEvent, Event
from ._task import TaskBase, TaskEvent


class Scheduler:
    """Manages task dependencies and schedules their execution.

    The scheduler processes tasks from a request queue. When an iterable of
    tasks is received, it resolves dependencies *within that sequence*. If a
    task's dependency is not found in the sequence, a `RuntimeError` is raised.
    A task is scheduled only after its dependencies complete successfully. If a
    dependency fails (finishes with an error), any tasks that depend on it are
    cancelled.

    After resolving dependencies, tasks are scheduled by placing them on an
    internal queue to be run. To limit the number of concurrent running tasks,
    an `asyncio.Semaphore` can be provided.

    Note:
        The request queue can be bounded. However, an iterable of tasks is
        treated as a single item. To avoid blocking the scheduler, you should
        limit the queue size and also ensure that submitted iterables are not
        too large, for instance by using smaller sequences or a generator.
    """

    def __init__(
        self,
        request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]],
        *,
        event_queue: asyncio.Queue[Event] | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Initializes the Scheduler.

        Args:
            request_queue: The queue from which to receive new tasks.
            event_queue:   An optional queue for unhandled exceptions.
            semaphore:     Optional semaphore to limit concurrent executions.

        Note:
            The `event_queue` is used to report unhandled exceptions that occur
            during scheduling, not for exceptions raised by the tasks.
        """
        self._request_queue = request_queue
        self._semaphore = semaphore
        self._event_queue = event_queue
        self._task_events_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._ready_queue: asyncio.Queue[TaskBase] = asyncio.Queue()
        self._requests: dict[uuid.UUID, TaskBase] = {}
        self._dependents: defaultdict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
        self._dependency_counts: defaultdict[uuid.UUID, int] = defaultdict(lambda: 0)
        self._request_handler: asyncio.Task[None] | None = None
        self._task_event_handler: asyncio.Task[None] | None = None
        self._task_submitter: asyncio.Task[None] | None = None

    def schedule(self) -> Self:
        """Starts the scheduler's background tasks.

        This method creates and starts the tasks for handling new task
        requests, processing state updates, and submitting ready tasks.

        Returns:
            The instance of the scheduler.
        """
        if self._request_handler is not None:
            return self
        self._request_handler = asyncio.create_task(self._handle_requests())
        self._request_handler.add_done_callback(self._done_callback)
        self._task_event_handler = asyncio.create_task(self._handle_task_events())
        self._task_event_handler.add_done_callback(self._done_callback)
        self._task_submitter = asyncio.create_task(self._submit_ready_tasks())
        self._task_submitter.add_done_callback(self._done_callback)
        return self

    async def cancel(self) -> None:
        """Cancels all running tasks and stops the scheduler.

        This method attempts to cancel all tasks currently managed by the
        scheduler and then stops the scheduler's own background tasks.
        """
        for task in list(self._requests.values()):
            with contextlib.suppress(Exception):
                await task.cancel()

        self._requests.clear()
        self._dependents.clear()
        self._dependency_counts.clear()

        if self._request_handler and not self._request_handler.done():
            self._request_handler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._request_handler
            self._request_handler = None
        if self._task_event_handler and not self._task_event_handler.done():
            self._task_event_handler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task_event_handler
            self._task_event_handler = None
        if self._task_submitter and not self._task_submitter.done():
            self._task_submitter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task_submitter
            self._task_submitter = None

    def _done_callback(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return

        def _error_fallback(exc: BaseException) -> None:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Unhandled task exception could not be queued",
                    "exception": exc,
                    "task": task,
                }
            )

        exc = task.exception()
        if exc is not None:
            if self._event_queue is None:
                _error_fallback(exc)
            else:
                try:
                    self._event_queue.put_nowait(
                        ErrorEvent(source="scheduler", error=exc)
                    )
                except Exception:  # noqa: BLE001
                    _error_fallback(exc)

    async def _handle_requests(self) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                ready: list[TaskBase] = []
                requests = await self._request_queue.get()
                if not isinstance(requests, Iterable):
                    requests = [requests]
                ready = self._handle_request_sequence(requests)
                for item in ready:
                    await self._ready_queue.put(item)

    async def _handle_task_events(self) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                event = await self._task_events_queue.get()
                if isinstance(event, TaskEvent):
                    ready = self._handle_one_task_event(event)
                    for item in ready:
                        await self._ready_queue.put(item)
                if self._event_queue is not None:
                    await self._event_queue.put(event)

    async def _submit_ready_tasks(self) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                item = await self._ready_queue.get()
                item.schedule(self._task_events_queue, semaphore=self._semaphore)

    def _handle_request_sequence(self, tasks: Iterable[TaskBase]) -> list[TaskBase]:
        ready: list[TaskBase] = []
        for task in tasks:
            if task.state != "pending":
                continue
            self._requests[task.tid] = task
            for tid in task.dependencies:
                if tid in self._requests:
                    self._dependents[tid].append(task.tid)
                    self._dependency_counts[task.tid] += 1
                else:
                    msg = f"Task dependency {tid} not found or invalid for task {task.tid}."
                    raise RuntimeError(msg)
            if not task.dependencies:
                ready.append(task)
        return ready

    def _handle_one_task_event(self, event: TaskEvent) -> list[TaskBase]:
        ready: list[TaskBase] = []
        assert isinstance(event.source, uuid.UUID)
        tid = event.source
        if event.state == "done":
            for dep in self._dependents[tid]:
                if event.error is None:
                    self._dependency_counts[dep] -= 1
                    if self._dependency_counts[dep] == 0:
                        ready.append(self._requests[dep])
                        self._dependency_counts.pop(dep, 0)
                else:
                    self._requests.pop(dep, None)
                    self._dependency_counts.pop(dep, 0)
        if event.state in {"done", "cancelled"}:
            self._dependents.pop(tid, [])
            self._requests.pop(tid, None)
        return ready
