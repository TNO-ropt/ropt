"""Base classes for the executors.

An executor runs work items on a concrete mechanism: worker threads, worker
processes, local jobs, or an HPC cluster.
[`run`][ropt.components.executors.Executor.run] blocks until every call it was
given has a result, so the thread that waits for the results is the thread that
does the work of getting them, and an executor owns no thread of its own.

Whatever a scope acquires, that scope releases. A batch acquires cluster jobs
and worker processes, and `run`'s `finally` releases them, so nothing a batch
started outlives it. An executor acquires its workers, and releases them when it
is collected, so there is nothing for a caller to close.

Reading the caller's own location to refuse a call that could only hang leaves
the meaning of an operation fixed by its arguments: a thread already running one
of the executor's work items turns a deadlock into an error, and does nothing
else.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ropt.exceptions import ExecutorStopped, WorkflowError

from ._picklable import picklable_exception

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.concurrency import AbortSignal

_ON_WORKER = (
    "This executor cannot be used from work that is already running on it: the "
    "caller would wait for the workers it is itself occupying, which deadlocks "
    "once they are all busy. Give the inner run an executor of its own, or none "
    "at all."
)

_STOPPED = "The executor can no longer run this work."

_NOT_RUN = "The work item was not run: the batch was stopped."


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
    """A work item was never attempted.

    Returned in the work item's position when the batch it belonged to was
    abandoned, so nothing about the item or the executor failed.

    Attributes:
        message: Why it was not attempted.
    """

    message: str


# The two outcomes that stand in for a result, wherever they are handled alike.
_NO_RESULT = (ExecutorFailure, WorkNotRun)


@dataclass(frozen=True)
class _Raised:
    # Marks an exception as *raised* by a call rather than returned by it, so
    # that a function returning an exception does not abort its batch.
    error: BaseException


def _calls(
    bundle: Sequence[WorkItem],
) -> list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]:
    return [(item.function, item.args, item.kwargs) for item in bundle]


def _run_bundle(
    calls: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
    *,
    sendable: bool = False,
) -> list[Any]:
    # Runs a whole bundle in one worker, so it must be reachable by name in a
    # worker process and in a job interpreter. A call that raises yields a
    # `_Raised` rather than taking its bundle mates with it; `sendable` is set
    # where the result has to survive serialization back to the caller.
    results: list[Any] = []
    for function, args, kwargs in calls:
        try:
            results.append(function(*args, **kwargs))
        except Exception as exc:  # ruff: ignore[blind-except]
            results.append(_Raised(picklable_exception(exc) if sendable else exc))
    return results


class Executor(ABC):
    """Abstract base class for executor components within an optimization workflow.

    An executor is constructed, run on, and dropped. Its workers are released
    when it is collected, so there is nothing to close.

    See [Error handling](../advanced/parallel.md#error-handling) for the
    distinction an implementation must make between an infrastructure failure,
    returned in the call's position as an
    [`ExecutorFailure`][ropt.components.executors.ExecutorFailure], and an
    exception from the work item's own function, which is re-raised in the
    caller.
    """

    @abstractmethod
    def run(
        self,
        calls: Sequence[WorkItem],
        *,
        bundle_size: int | None = None,
        collect_errors: bool = False,
        abort_signal: AbortSignal | None = None,
    ) -> list[Any]:
        """Run the calls and return their results, in the order they were given.

        Blocks until every call has a result. A call the machinery could not run
        gets an [`ExecutorFailure`][ropt.components.executors.ExecutorFailure]
        in its position, and a call that was never attempted gets a
        [`WorkNotRun`][ropt.components.executors.WorkNotRun].

        An exception raised by a call's own function is re-raised here, without
        waiting for the rest of the batch. With `collect_errors` it is placed in
        the call's position instead, and every call keeps its place.

        An `abort_signal` that fires abandons the batch: queued calls are never
        started, started ones are cancelled where the mechanism allows it, and
        each abandoned call gets a `WorkNotRun`. Calls already running on a
        worker thread or process run to their end.

        May be called from any thread, except one of the executor's own workers:
        that caller would wait for workers it is itself occupying, so it is
        refused rather than left to deadlock.

        Whatever this batch started is released before this returns, however it
        returns.

        Args:
            calls:          The work items to run.
            bundle_size:    Calls per worker task, `0` for all of them.
            collect_errors: Whether an exception is returned instead of raised.
            abort_signal:   An optional signal that abandons the batch.

        Returns:
            One result per call, in the order of `calls`.

        Raises:
            WorkflowError:   If the caller is one of the executor's workers.
            ExecutorStopped: If the executor could no longer run the work.
        """


class _ThreadState(threading.local):
    # A `threading.local` attribute exists per thread that assigns it; every
    # other thread reads the class attribute below.
    running_work_item: bool = False


class ExecutorBase(Executor):
    """A base class for executors.

    Handles the refusal of a caller that is one of the workers, and the split of
    a batch into bundles. Subclasses implement `_run_bundles`, which must
    release whatever the batch started before it returns.

    A subclass whose workers are threads in this process sets
    `_thread_state.running_work_item` for as long as a work item runs on one,
    and `run` refuses a caller that has it set. Workers in another interpreter
    cannot reach the executor, so it stays `False` there.
    """

    def __init__(self, *, bundle_size: int = 1) -> None:
        """Initialize the executor.

        Args:
            bundle_size: Calls per worker task, `0` for a whole batch.

        Raises:
            ValueError: If `bundle_size` is negative.
        """  # ruff: ignore[docstring-extraneous-exception]
        super().__init__()
        _check_bundle_size(bundle_size)
        self._default_bundle_size = bundle_size
        self._thread_state = _ThreadState()

    def run(
        self,
        calls: Sequence[WorkItem],
        *,
        bundle_size: int | None = None,
        collect_errors: bool = False,
        abort_signal: AbortSignal | None = None,
    ) -> list[Any]:
        """Run the calls and return their results, in the order they were given.

        Args:
            calls:          The work items to run.
            bundle_size:    Calls per worker task, `0` for all of them.
            collect_errors: Whether an exception is returned instead of raised.
            abort_signal:   An optional signal that abandons the batch.

        Returns:
            One result per call, in the order of `calls`.

        Raises:
            WorkflowError: If the caller is one of the executor's workers.
        """
        if self._thread_state.running_work_item:
            raise WorkflowError(_ON_WORKER)
        items = list(calls)
        if not items:
            return []
        size = self._resolve_bundle_size(len(items), bundle_size)
        bundles = [items[start : start + size] for start in range(0, len(items), size)]
        results: list[Any] = [None] * len(items)
        stored: set[int] = set()

        def store(index: int, bundle_result: Any) -> None:  # ruff: ignore[any-type]
            stored.add(index)
            _store(
                results,
                index * size,
                len(bundles[index]),
                bundle_result,
                collect_errors=collect_errors,
            )

        self._run_bundles(bundles, store, abort_signal)
        # A batch that was abandoned leaves its remaining slots empty, and an
        # empty slot is indistinguishable from a call that returned None.
        for index, bundle in enumerate(bundles):
            if index not in stored:
                offset = index * size
                results[offset : offset + len(bundle)] = [WorkNotRun(_NOT_RUN)] * len(
                    bundle
                )
        return results

    def _resolve_bundle_size(self, count: int, bundle_size: int | None) -> int:
        if bundle_size is None:
            size = self._default_bundle_size
        else:
            _check_bundle_size(bundle_size)
            size = bundle_size
        # A bundle never spans batches, so zero is the whole batch.
        return count if size == 0 else size

    @abstractmethod
    def _run_bundles(
        self,
        bundles: list[list[WorkItem]],
        store: Callable[[int, Any], None],
        abort_signal: AbortSignal | None,
    ) -> None:
        """Run the bundles, passing each one's result to `store` as it arrives.

        `store` takes the index of the bundle in `bundles` and either the list
        of its results or an
        [`ExecutorFailure`][ropt.components.executors.ExecutorFailure] for the
        whole bundle. It may raise, which is how a user-code exception reaches
        the caller: whatever this batch started must be released before that
        exception leaves.

        An `abort_signal` that fires must bring this call back promptly, leaving
        the bundles it did not store for the caller to mark as not run.

        Args:
            bundles:      The bundles to run.
            store:        Callback taking a bundle index and its result.
            abort_signal: An optional signal that abandons the batch.
        """


def _check_bundle_size(bundle_size: int) -> None:
    if bundle_size < 0:
        msg = f"bundle_size must be >= 0, got {bundle_size}"
        raise ValueError(msg)


def _store(
    results: list[Any],
    offset: int,
    count: int,
    bundle_result: Any,  # ruff: ignore[any-type]
    *,
    collect_errors: bool,
) -> None:
    if not isinstance(bundle_result, _NO_RESULT) and (
        not isinstance(bundle_result, list) or len(bundle_result) != count
    ):
        bundle_result = ExecutorFailure(
            f"A job returned a result that does not match its {count} work item(s)."
        )
    if isinstance(bundle_result, _NO_RESULT):
        results[offset : offset + count] = [bundle_result] * count
        return
    if not collect_errors:
        # Raised as the bundle arrives, so a failure still does not wait for
        # the rest of the batch.
        for item in bundle_result:
            if isinstance(item, _Raised):
                raise item.error
    results[offset : offset + count] = [
        item.error if isinstance(item, _Raised) else item for item in bundle_result
    ]


def _stopped() -> ExecutorStopped:
    return ExecutorStopped(_STOPPED)
