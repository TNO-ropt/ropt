"""Base classes for the executors.

An executor runs work items on a concrete mechanism: worker threads, worker
processes, local jobs, or an HPC cluster.
[`run`][ropt.components.executors.Executor.run] blocks until every call it was
given has a result, so the thread that waits for the results is the thread that
does the work of getting them, and an executor owns no thread of its own.

Whatever a scope acquires, that scope releases. A batch acquires cluster jobs
and worker processes, and `run`'s `finally` releases them, so nothing a batch
started outlives it. The exception is a call already running on a worker thread
or process when the batch is aborted: it cannot be interrupted, so it runs
after `run` has returned, until it notices the abort or ends. An executor
acquires its workers, and releases them when it is collected, so there is
nothing for a caller to close.

Reading the caller's context to refuse a call that could only hang leaves the
meaning of an operation fixed by its arguments: code running one of the
executor's work items, or a thread started from it with a copy of its context,
turns a deadlock into an error, and does nothing else.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from ropt._logging import get_logger
from ropt.exceptions import AbortedError, ExecutorStopped, WorkflowError

from ._picklable import picklable_exception

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable, Sequence
    from concurrent.futures import Future

    from ropt.components.concurrency import AbortScope

_logger = get_logger(__name__)

_ON_WORKER = (
    "This executor cannot be used from work that is already running on it: the "
    "caller would wait for the workers it is itself occupying, which deadlocks "
    "once they are all busy. Give the inner task an executor of its own, or none "
    "at all."
)

_STOPPED = "The executor can no longer run this work."

_NOT_RUN = "The work item has no result: the batch was aborted."

_LEFT_RUNNING = "A work item that was still running when its batch returned raised: %r"

# How long a bounded wait may block before Python gets to run a pending signal
# handler, which some extension modules otherwise hold off until the wait ends.
_POLL_INTERVAL = 0.1


@dataclass(kw_only=True)
class WorkItem:
    """A single unit of work to run on a worker.

    A work item is a plain description of a call, so it can be handed to a
    worker process on its own.

    Attributes:
        function: The function to execute.
        args:     The arguments to pass to the function.
        kwargs:   The keyword arguments to pass to the function.
    """

    function: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutorFailure:
    """The executor could not run a work item.

    Returned in the work item's position rather than raised, so the executor
    stays available for further work.

    Attributes:
        message: What went wrong.
    """

    message: str


@dataclass(frozen=True)
class WorkNotRun:
    """A work item has no result because its batch was aborted.

    Returned in the work item's position when the batch it belonged to was
    aborted before the item finished: it was never started, it was cancelled,
    or it was still running when the batch returned. Nothing about the item or
    the executor failed.

    Attributes:
        message: Why it has no result.
    """

    message: str


@dataclass(frozen=True)
class _Raised:
    # Marks an exception as *raised* by a call rather than returned by it, so
    # that a function returning an exception does not abort its batch.
    error: BaseException


def _run_item(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    sendable: bool = False,
) -> Any:  # ruff: ignore[any-type]
    try:
        return function(*args, **kwargs)
    except Exception as exc:  # ruff: ignore[blind-except]
        return _Raised(picklable_exception(exc) if sendable else exc)


def _log_left_running(
    outcome: Callable[[Future[Any]], Any], future: Future[Any]
) -> None:
    # Work that its batch no longer waits for has no caller left to raise into.
    # An abort it passed on is the batch's own abort, not an error of the work.
    try:
        result = outcome(future)
    except Exception as exc:  # ruff: ignore[blind-except]
        result = _Raised(exc)
    if isinstance(result, _Raised) and not isinstance(result.error, AbortedError):
        _logger.warning(_LEFT_RUNNING, result.error, exc_info=result.error)


class Executor(ABC):
    """Abstract base class for executor components within an optimization workflow.

    An executor is constructed, run on, and dropped. Its workers are released
    when it is collected, so there is nothing to close.

    See
    [When an evaluation fails](../running/parallel.md#when-an-evaluation-fails)
    for the distinction an implementation must make between an infrastructure
    failure, returned in the call's position as an
    [`ExecutorFailure`][ropt.components.executors.ExecutorFailure], and an
    exception from the work item's own function, which is re-raised in the
    caller.
    """

    @abstractmethod
    def run(
        self,
        calls: Sequence[WorkItem],
        *,
        collect_errors: bool = False,
        abort_scope: AbortScope | None = None,
    ) -> list[Any]:
        """Run the calls and return their results, in the order they were given.

        Blocks until every call has a result, or until `abort_scope` aborts. A
        call the machinery could not run gets an
        [`ExecutorFailure`][ropt.components.executors.ExecutorFailure] in its
        position, and a call left without a result by an abort gets a
        [`WorkNotRun`][ropt.components.executors.WorkNotRun].

        An exception raised by a call's own function is re-raised here, without
        waiting for the rest of the batch. With `collect_errors` it is placed in
        the call's position instead, and every call keeps its place.

        When `abort_scope` aborts, the batch is aborted: queued calls are never
        started, started ones are cancelled where the mechanism allows it, and
        `run` returns without waiting for the rest. Each call that did not
        finish gets a `WorkNotRun`. A call already running on a worker thread
        or process cannot be interrupted, and neither can a call the mechanism
        has handed to a worker but can no longer withdraw: both keep running
        after `run` has returned, and an exception they raise is logged as a
        warning. An executor that runs work items in this process, or in its
        own worker processes, runs each inside an abort scope that `abort_scope`
        aborts, so such a call can end early. See
        [Stopping a run](../running/parallel.md#stopping-a-run).

        May be called from any thread, except from work running on one of the
        executor's own workers, or from a thread started by that work with a
        copy of its context: that caller would wait for workers it is itself
        occupying, so it is refused rather than left to deadlock.

        Whatever this batch started is released before this returns, however it
        returns.

        Args:
            calls:          The work items to run.
            collect_errors: Whether an exception is returned instead of raised.
            abort_scope:    An optional scope that aborts the batch.

        Returns:
            One result per call, in the order of `calls`.

        Raises:
            WorkflowError:   If the call comes from one of the executor's workers.
            ExecutorStopped: If the executor could no longer run the work.
        """


# A context variable, so that threads started with a copy of the context inherit it.
_running_on: ContextVar[Executor | None] = ContextVar("running_on", default=None)


class ExecutorBase(Executor):
    """A base class for executors.

    Handles the refusal of a caller that is one of the workers, and the split of
    a batch into individual tasks. Subclasses implement `_run_items`, which must
    release whatever the batch started before it returns.

    A subclass whose workers are threads in this process sets `_running_on` to
    the executor for as long as a work item runs on one, and `run` refuses a
    caller whose context has it set, including a thread started with a copy of
    that context. Workers in another interpreter cannot reach the executor, so
    it is never set there. Such a subclass also runs the work item inside the
    [`as_current`][ropt.components.concurrency.AbortScope.as_current] block of
    the `abort_scope` that `run` was given, so that a task started from the
    work item takes the one that sent it as its parent.
    """

    def __init__(self) -> None:
        """Initialize the executor."""
        super().__init__()

    def _stop(self) -> None:
        # Called when the owner releases the executor, after aborting its work:
        # refuse new work and start ending what runs, without waiting.
        pass

    def _close(self, deadline: float) -> None:
        # Called after `_stop`: wait until `deadline` at most for running work.
        pass

    def _retire(self) -> None:
        # Called after the owner aborted the work: end what still runs once a
        # grace period has passed, without waiting for it here.
        pass

    @override
    def run(
        self,
        calls: Sequence[WorkItem],
        *,
        collect_errors: bool = False,
        abort_scope: AbortScope | None = None,
    ) -> list[Any]:
        """Run the calls and return their results, in the order they were given.

        Args:
            calls:          The work items to run.
            collect_errors: Whether an exception is returned instead of raised.
            abort_scope:    An optional scope that aborts the batch.

        Returns:
            One result per call, in the order of `calls`.

        Raises:
            WorkflowError: If the call comes from one of the executor's workers.
        """
        if _running_on.get() is self:
            raise WorkflowError(_ON_WORKER)
        items = list(calls)
        if not items:
            return []
        results: list[Any] = [None] * len(items)
        stored: set[int] = set()

        def store(index: int, result: Any) -> None:  # ruff: ignore[any-type]
            stored.add(index)
            if isinstance(result, _Raised):
                if not collect_errors:
                    raise result.error
                result = result.error
            results[index] = result

        self._run_items(items, store, abort_scope)
        # An aborted batch leaves the slots of the calls that did not run empty,
        # and an empty slot is indistinguishable from a call that returned None.
        for index in range(len(items)):
            if index not in stored:
                results[index] = WorkNotRun(_NOT_RUN)
        return results

    @abstractmethod
    def _run_items(
        self,
        items: list[WorkItem],
        store: Callable[[int, Any], None],
        abort_scope: AbortScope | None,
    ) -> None:
        """Run the items, passing each result to `store` as it arrives.

        `store` takes the index of an item in `items` and its result or an
        [`ExecutorFailure`][ropt.components.executors.ExecutorFailure].
        It may raise, which is how a user-code exception reaches
        the caller: whatever this batch started must be released before that
        exception leaves.

        An `abort_scope` that aborts must bring this call back promptly, leaving
        the items it did not store for the caller to mark as not run.

        Args:
            items:       The work items to run.
            store:       Callback taking an item index and its result.
            abort_scope: An optional scope that aborts the batch.
        """


def _stopped() -> ExecutorStopped:
    return ExecutorStopped(_STOPPED)


def _wait_for(
    condition: threading.Condition, predicate: Callable[[], bool], deadline: float
) -> bool:
    with condition:
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            condition.wait(min(remaining, _POLL_INTERVAL))
    return True
