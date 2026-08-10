"""Defines base classes for asynchronous executors."""

from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class Executor(ABC):
    """Abstract base class for executor components within an optimization workflow.

    Subclasses must implement the following abstract methods and properties:

    - [`start`][ropt.components.executors.Executor.start]: Starts the executor.
    - [`cancel`][ropt.components.executors.Executor.cancel]: Stops the executor.
    - [`task_queue`][ropt.components.executors.Executor.task_queue]: Retrieves the
      executor's task queue.
    - [`loop`][ropt.components.executors.Executor.loop]: Retrieves the
      currently running asyncio loop.
    - [`task_group`][ropt.components.executors.Executor.task_group]: The asyncio.Taskgroup
      used by this executor.
    - [`is_running`][ropt.components.executors.Executor.is_running]: Checks if the
      executor is running.
    """

    @property
    @abstractmethod
    def task_queue(self) -> asyncio.Queue[Any]:
        """The task queue."""

    @property
    @abstractmethod
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The asyncio loop used by this executor."""

    @property
    @abstractmethod
    def task_group(self) -> asyncio.TaskGroup | None:
        """The task group used by this executor."""

    @abstractmethod
    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.

        Raises:
            RuntimeError: If the executor is already running or using an
                          external queue.
        """

    @abstractmethod
    def cancel(self) -> None:
        """Stop the executor."""

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the executor is running.

        Returns:
            True if the executor is running.
        """


class ExecutorBase(Executor):
    """A base class for asynchronous executors."""

    def __init__(self, queue_size: int = 0) -> None:
        """Initialize the executor.

        Arguments:
            queue_size: Maximum size of the task queue.
        """
        super().__init__()
        self._task_queue: asyncio.Queue[Task] = asyncio.Queue(queue_size)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._running = threading.Event()
        self._ready_event = asyncio.Event()
        self._wait_event = asyncio.Event()
        self._wait_task: asyncio.Task[None] | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The asyncio loop used by this executor."""
        return self._loop

    @property
    def task_group(self) -> asyncio.TaskGroup | None:
        """The task group used by this executor."""
        return self._task_group

    @property
    def task_queue(self) -> asyncio.Queue[Task]:
        """The task queue."""
        return self._task_queue

    async def _finish_start(self, task_group: asyncio.TaskGroup) -> None:
        if self._running.is_set():
            msg = "Executor is already running."
            raise RuntimeError(msg)
        self._running.set()
        self._loop = asyncio.get_running_loop()
        self._task_group = task_group
        self._ready_event.clear()
        self._wait_event.clear()
        self._wait_task = task_group.create_task(self._wait_for_cancel())
        await self._ready_event.wait()

    async def _wait_for_cancel(self) -> None:
        self._ready_event.set()
        try:
            await self._wait_event.wait()
        finally:
            if self._running.is_set():
                self._running.clear()
                self.cleanup()

    def cancel(self) -> None:
        """Stop the executor."""
        if self._wait_task is not None:
            self._wait_task.cancel()
            self._wait_task = None

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up the executor."""

    def is_running(self) -> bool:
        """Check if the executor is running.

        Returns:
            True if the executor is running.
        """
        return self._running.is_set()

    def _drain_and_kill(self) -> None:
        """Drain the task queue and kill clients."""
        while not self._task_queue.empty():
            try:
                task = self._task_queue.get_nowait()
                task.cancel_all()
                self._task_queue.task_done()
            except asyncio.QueueEmpty:
                break


@dataclass(kw_only=True)
class Task(ABC):
    """A task to be executed by a worker.

    Task results are delivered on the associated
    [`ResultsQueue`][ropt.components.executors.ResultsQueue]. Two distinct
    failure classes are distinguished, following the error contract described in
    [Parallel Evaluation](../workflows/parallel.md#error-handling):

    - An **infrastructure failure** (a killed worker process, or missing/corrupt
      HPC output) is delivered as an ordinary result whose value is an
      [`ExecutorFailure`][ropt.exceptions.ExecutorFailure] via
      [`put_result`][ropt.components.executors.Task.put_result]. This is a
      tolerated per-realization failure.
    - A **user-code exception** (the task function itself raises) is delivered
      via [`put_error`][ropt.components.executors.Task.put_error], which places
      the exception on the queue and closes it. The owning evaluator re-raises
      the original exception unchanged, aborting the current evaluation; the
      executor keeps running.

    Attributes:
        function:      The function to execute.
        args:          The arguments to pass to the function.
        kwargs:        The keyword arguments to pass to the function.
        results_queue: The queue to put the result in.
        result:        The result of the function, or None if no result is available.
        name:          Optional unique name of the task.
    """

    function: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    results_queue: ResultsQueue
    result: Any | None = None
    name: str | None = None

    def put_result(self, result: Any) -> None:  # ruff: ignore[any-type]
        """Put the result in the result queue."""
        self.result = result
        self.results_queue.put(self)

    def put_error(self, exc: BaseException) -> None:
        """Deliver a user-code exception on the result queue.

        Places the exception raised by the task's function on the queue and
        closes it. Like [`cancel_all`][ropt.components.executors.Task.cancel_all]
        this unblocks the waiting evaluator, but it additionally carries the
        exception so the evaluator can re-raise the original unchanged.

        Args:
            exc: The exception raised by the task's function.
        """
        self.results_queue.put(exc)
        self.results_queue.close()

    def cancel_all(self) -> None:
        """Stop putting results in the result queue."""
        self.results_queue.put(None)
        self.results_queue.close()


class ResultsQueue(queue.Queue["Task | BaseException | None"]):
    """A queue that can be closed.

    Items delivered on this queue follow the error contract of
    [`Task`][ropt.components.executors.Task]: a [`Task`][ropt.components.executors.Task]
    carries a normal result (including an
    [`ExecutorFailure`][ropt.exceptions.ExecutorFailure] as its `result`), a
    `BaseException` signals a user-code exception that must abort the
    evaluation, and `None` is a plain sentinel used to unblock a waiting
    consumer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        """Initialize the queue."""
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self) -> None:
        """Close the queue."""
        self.closed = True

    def put(self, item: Task | BaseException | None, *args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        """Put an item in the queue."""
        if not self.closed:
            super().put(item, *args, **kwargs)
