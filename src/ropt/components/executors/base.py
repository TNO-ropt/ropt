"""Defines base classes for asynchronous executors.

An executor runs on an event loop, while the code waiting for its results sits
in a thread. The two meet on a [`Submission`][ropt.components.executors.Submission]:
the caller blocks on its results queue, and everything the executor itself does
runs on the loop thread, which is why its bookkeeping needs no locking. The few
flags that do cross the boundary are `threading.Event`s.

Handing over a submission transfers responsibility for it: whatever happens, the
executor either runs its work items or ends the submission, so the caller is
released.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ropt.components._loop import on_loop_thread, schedule
from ropt.components._transferred import _make_placeholder
from ropt.exceptions import ExecutorStopped, WorkflowError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import NoReturn


@dataclass(kw_only=True)
class WorkItem:
    """A single unit of work to run on a worker.

    A work item is a plain description of a call. It carries no delivery
    channel, so it can be handed to a worker process without dragging the
    submission that owns it along.

    Attributes:
        function: The function to execute.
        args:     The arguments to pass to the function.
        kwargs:   The keyword arguments to pass to the function.
        result:   The result of the function, only meaningful once the work
                  item has been delivered.
        name:     Optional unique name of the work item.
    """

    function: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    name: str | None = None


class _ResultsQueue(queue.Queue["WorkItem | BaseException | None"]):
    # Results are delivered from the event loop, so putting must never block:
    # this queue is deliberately unbounded and cannot be configured otherwise.

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def put(
        self,
        item: WorkItem | BaseException | None,
        *args: Any,  # ruff: ignore[any-type]
        **kwargs: Any,  # ruff: ignore[any-type]
    ) -> None:
        # Silently dropped once closed, which makes ending a submission twice
        # harmless: the first end is the one the caller sees.
        if not self.closed:
            super().put(item, *args, **kwargs)


class Submission:
    """A group of work items and the channel back to the caller awaiting them.

    A submission owns its results channel, so ending it is one operation on one
    object. Handing a submission to an executor transfers responsibility: the
    executor either runs the work items or aborts the submission, so a caller
    blocked in [`collect`][ropt.components.executors.Submission.collect] is
    always released.

    See [Error handling](../workflows/parallel.md#error-handling) for how
    infrastructure failures (delivered via
    [`deliver`][ropt.components.executors.Submission.deliver]) and user-code
    exceptions (ended via [`fail`][ropt.components.executors.Submission.fail])
    are distinguished.
    """

    def __init__(self, work_items: Sequence[WorkItem]) -> None:
        """Initialize the submission.

        Args:
            work_items: The work items to run.
        """
        self._work_items = list(work_items)
        self._results = _ResultsQueue()
        self._outstanding = len(self._work_items)
        self._ended = False

    @property
    def work_items(self) -> list[WorkItem]:
        """The work items to run.

        Returns:
            The work items.
        """
        return self._work_items

    @property
    def is_finished(self) -> bool:
        """Whether anything more will be delivered.

        Returns:
            `True` if every work item was delivered, or the submission ended.
        """
        return self._ended or self._outstanding <= 0

    def deliver(self, work_item: WorkItem, result: Any) -> None:  # ruff: ignore[any-type]
        """Deliver the result of a single work item.

        Args:
            work_item: The work item that ran.
            result:    The result it produced.
        """
        work_item.result = result
        self._results.put(work_item)
        self._outstanding -= 1

    def fail(self, exc: BaseException) -> None:
        """End the submission, re-raising an exception in the caller.

        Args:
            exc: The exception raised by the work item's function.
        """
        self._results.put(exc)
        self._end()

    def abort(self) -> None:
        """End the submission, releasing the caller with `ExecutorStopped`."""
        self._results.put(None)
        self._end()

    def _end(self) -> None:
        self._results.close()
        self._ended = True

    def collect(self, on_result: Callable[[WorkItem], None]) -> None:
        """Wait for every work item and pass each finished one to `on_result`.

        The submission is ended if this returns early, including when
        `on_result` itself raises, so the executor never keeps delivering to a
        caller that has left.

        Args:
            on_result: Callback invoked with each finished work item.

        Raises:
            ExecutorStopped: If the submission ended before every result was
                             delivered.
        """  # ruff: ignore[docstring-extraneous-exception]
        try:
            self._drain(on_result)
        except BaseException:
            self._end()
            raise

    def _drain(self, on_result: Callable[[WorkItem], None]) -> None:
        # One `get` per work item: the count is what ends the loop, since a
        # finished submission sends no sentinel of its own.
        for _ in range(len(self._work_items)):
            item = self._results.get()
            if item is None:
                self._raise_stopped()
            if isinstance(item, BaseException):
                raise item
            on_result(item)

    def _raise_stopped(self) -> NoReturn:
        # Prefer a real exception over the generic abort if one is queued too.
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, BaseException):
                raise item
        msg = "The worker pool was closed."
        raise ExecutorStopped(msg)


class Executor(ABC):
    """Abstract base class for executor components within an optimization workflow.

    Subclasses must implement the following abstract methods:

    - [`start`][ropt.components.executors.Executor.start]: Starts the executor.
    - [`cancel`][ropt.components.executors.Executor.cancel]: Stops the executor.
    - [`submit`][ropt.components.executors.Executor.submit]: Hands over a submission.
    - [`is_running`][ropt.components.executors.Executor.is_running]: Reports
      whether the executor accepts work.
    """

    def __reduce__(self) -> tuple[object, tuple[str]]:  # ruff: ignore[undocumented-magic-method]
        # An executor drives workers from the process that started it, so it
        # cannot follow work into one. It arrives there as a placeholder, which
        # the worker reports by name.
        return (_make_placeholder, ("An executor",))

    @abstractmethod
    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.

        Raises:
            WorkflowError: If the executor is already running.
        """

    @abstractmethod
    def cancel(self) -> None:
        """Stop the executor.

        May be called from any thread.
        """

    def on_worker_loop(self) -> bool:  # ruff: ignore[no-self-use]
        """Report whether the caller is on the event loop that runs the work.

        Blocking that loop starves the work being waited for, so callers use
        this to refuse rather than deadlock. Implementations that run their
        work on an event loop must override this; the default `False` is for
        executors that do not have one.

        Returns:
            `True` if the calling thread is running the executor's loop.
        """
        return False

    def on_worker_thread(self) -> bool:  # ruff: ignore[no-self-use]
        """Report whether the caller is running as one of this executor's workers.

        Such a caller occupies a worker for as long as it waits, so work it
        submits here can only start once it stops waiting. Implementations whose
        workers run in this process must override this; the default `False` is
        for executors whose workers cannot submit back in the first place, and
        leaves the refusal in `submit` inactive.

        Returns:
            `True` if the calling thread is running this executor's work.
        """
        return False

    @abstractmethod
    def is_running(self) -> bool:
        """Report whether the executor accepts work.

        May be called from any thread. A `False` result means a submission would
        be aborted rather than run, so a caller that is able to do the work
        itself may fall back to doing so.

        Returns:
            `True` if the executor accepts work, `False` otherwise.
        """

    @abstractmethod
    def submit(self, submission: Submission) -> None:
        """Hand a submission to the executor.

        May be called from any thread, except one of the executor's own workers:
        that caller waits for workers it is itself occupying, so it is refused
        rather than left to deadlock. A submission handed to an executor that
        is no longer running is aborted rather than queued, so its caller is
        never left waiting for results that cannot arrive.

        Args:
            submission: The submission to run.

        Raises:
            WorkflowError: If called from one of this executor's workers.
        """


class ExecutorBase(Executor):
    """A base class for asynchronous executors.

    Owns every submission it accepts, so stopping the executor releases all
    waiting callers from a single place.

    Implementations must call `_begin_start` before creating any resources, and
    `_finish_start` once they are in place.
    """

    def __init__(self) -> None:
        """Initialize the executor."""
        super().__init__()
        self._work_queue: asyncio.Queue[tuple[Submission, WorkItem]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task_group: asyncio.TaskGroup | None = None
        # An Event, not a bool: read from whatever thread submits, asks whether
        # the executor runs, or cancels it.
        self._running = threading.Event()
        self._ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        # Touched only on the loop thread, so it needs no lock.
        self._submissions: set[Submission] = set()

    def submit(self, submission: Submission) -> None:
        """Hand a submission to the executor.

        Args:
            submission: The submission to run.

        Raises:
            WorkflowError: If called from one of this executor's workers.
        """
        if self.on_worker_thread():
            msg = (
                "This worker pool cannot be used from work that is already "
                "running on it: the caller would wait for the workers it is "
                "itself occupying, which deadlocks once they are all busy. "
                "Give the inner run its own pool, or a serial pool."
            )
            raise WorkflowError(msg)
        if not self._running.is_set() or not schedule(
            self._loop, self._accept, submission
        ):
            # Either already stopped, or the loop refused the callback while
            # stopping. `_accept` checks again on the loop thread, which is the
            # check that settles the race.
            submission.abort()

    def is_running(self) -> bool:
        """Report whether the executor accepts work.

        Returns:
            `True` if the executor accepts work, `False` otherwise.
        """
        return self._loop is not None and self._running.is_set()

    def on_worker_loop(self) -> bool:
        """Report whether the caller is on the event loop that runs the work.

        Returns:
            `True` if the calling thread is running the executor's loop.
        """
        return on_loop_thread(self._loop)

    def _accept(self, submission: Submission) -> None:
        # On the loop thread.
        if not self._running.is_set():
            submission.abort()
            return
        if submission in self._submissions:
            return
        if submission.is_finished:
            # Its caller has already left, so there is nothing to deliver to.
            return
        self._submissions.add(submission)
        for work_item in submission.work_items:
            self._work_queue.put_nowait((submission, work_item))

    def _begin_start(self) -> None:
        """Guard against starting twice, before any resources are created.

        Raises:
            WorkflowError: If the executor is already running.
        """
        if self._running.is_set():
            msg = "The executor is already running."
            raise WorkflowError(msg)
        # Recreated per start: asyncio primitives bind to the loop that first
        # uses them, and a restart may well be on a different loop.
        self._work_queue = asyncio.Queue()
        self._ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()

    async def _finish_start(self, task_group: asyncio.TaskGroup) -> None:
        self._loop = asyncio.get_running_loop()
        self._task_group = task_group
        self._running.set()
        # The task group owns the task; stopping goes through the stop event.
        task_group.create_task(self._wait_for_cancel())
        await self._ready_event.wait()

    async def _wait_for_cancel(self) -> None:
        # Cleanup belongs on the loop thread, and this task is where it runs:
        # `cancel` only sets the event, from wherever it is called.
        self._ready_event.set()
        try:
            await self._stop_event.wait()
        finally:
            if self._running.is_set():
                self._running.clear()
                self._cleanup()
                self._loop = None
                self._task_group = None

    def cancel(self) -> None:
        """Stop the executor.

        May be called from any thread.
        """
        schedule(self._loop, self._stop_event.set)

    @abstractmethod
    def _cleanup(self) -> None:
        """Clean up the executor."""

    def _deliver(
        self,
        submission: Submission,
        work_item: WorkItem,
        result: Any,  # ruff: ignore[any-type]
    ) -> None:
        submission.deliver(work_item, result)
        if submission.is_finished:
            self._submissions.discard(submission)

    def _fail(self, submission: Submission, exc: BaseException) -> None:
        submission.fail(exc)
        self._submissions.discard(submission)

    def _abort(self, submission: Submission) -> None:
        submission.abort()
        self._submissions.discard(submission)

    def _cleanup_submissions(self) -> None:
        for submission in self._submissions:
            submission.abort()
        self._submissions.clear()
        # Drop the queued work too: its submissions have just been released,
        # and a restart begins with a fresh queue anyway.
        while not self._work_queue.empty():
            self._work_queue.get_nowait()
