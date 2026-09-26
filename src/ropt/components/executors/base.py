"""Base classes for the executors.

An executor runs work items on a concrete mechanism: worker threads, worker
processes, local jobs, or an HPC cluster.
[`run`][ropt.components.executors.Executor.run] blocks until every call it was
given has a result, so the thread that waits for the results is the thread that
does the work of getting them, and an executor owns no thread of its own.

Whatever a scope acquires, that scope releases. A batch acquires cluster jobs
and worker processes, and `run`'s `finally` releases them, so nothing a batch
started outlives it. An executor acquires its workers, and
[`close`][ropt.components.executors.Executor.close] releases them.

Reading the caller's own location to refuse a call that could only hang leaves
the meaning of an operation fixed by its arguments: a thread already running one
of the executor's work items turns a deadlock into an error, and does nothing
else.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from ropt.exceptions import ExecutorStopped, WorkflowError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_ON_WORKER = (
    "This executor cannot be used from work that is already running on it: the "
    "caller would wait for the workers it is itself occupying, which deadlocks "
    "once they are all busy. Give the inner run an executor of its own, or none "
    "at all."
)

_CLOSED = "This executor is closed and cannot take new work; build a new one."

_STOPPED = "The executor was closed while the work was running."


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


def _calls(
    bundle: Sequence[WorkItem],
) -> list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]:
    return [(item.function, item.args, item.kwargs) for item in bundle]


def _run_bundle(
    calls: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
) -> list[Any]:
    # Runs a whole bundle in one worker, so it must be reachable by name in a
    # worker process and in a job interpreter.
    return [function(*args, **kwargs) for function, args, kwargs in calls]


class Executor(ABC):
    """Abstract base class for executor components within an optimization workflow.

    An executor is used like a file: construct it, run work on it, close it.
    Closing is final, and releases the workers.

    See [Error handling](../advanced/parallel.md#error-handling) for the
    distinction an implementation must make between an infrastructure failure,
    returned in the call's position as an
    [`ExecutorFailure`][ropt.components.executors.ExecutorFailure], and an
    exception from the work item's own function, which is re-raised in the
    caller.
    """

    @abstractmethod
    def run(
        self, calls: Sequence[WorkItem], *, bundle_size: int | None = None
    ) -> list[Any]:
        """Run the calls and return their results, in the order they were given.

        Blocks until every call has a result. A call the machinery could not run
        gets an [`ExecutorFailure`][ropt.components.executors.ExecutorFailure]
        in its position; an exception raised by a call's own function is
        re-raised here, without waiting for the rest of the batch.

        May be called from any thread, except one of the executor's own workers:
        that caller would wait for workers it is itself occupying, so it is
        refused rather than left to deadlock.

        Whatever this batch started is released before this returns, however it
        returns.

        Args:
            calls:       The work items to run.
            bundle_size: Calls per worker task, `0` for all of them.

        Returns:
            One result per call, in the order of `calls`.

        Raises:
            WorkflowError:   If the executor is closed, or the caller is one of
                             its workers.
            ExecutorStopped: If the executor was closed while the work ran.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the executor's workers.

        Closing is final and may be called more than once. A caller blocked in
        [`run`][ropt.components.executors.Executor.run] is released with
        [`ExecutorStopped`][ropt.exceptions.ExecutorStopped].

        May be called from any thread.
        """

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether the executor has been closed.

        Returns:
            `True` once the executor was closed.
        """

    def __enter__(self) -> Self:
        """Enter a block that closes the executor on exit.

        Returns:
            The executor itself.
        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the executor."""
        self.close()


class _ThreadState(threading.local):
    # A `threading.local` attribute exists per thread that assigns it; every
    # other thread reads the class attribute below.
    running_work_item: bool = False


class ExecutorBase(Executor):
    """A base class for executors.

    Handles the lifecycle, the refusal of a caller that is one of the workers,
    and the split of a batch into bundles. Subclasses implement `_run_bundles`,
    which must release whatever the batch started before it returns, and
    `_release`, which releases the executor's own resources.

    The lock is available to subclasses as `_lock`: an implementation with
    shared state of its own keeps it under the same lock, so that closing and
    that state cannot be observed out of step.

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
        self._lock = threading.Lock()
        self._closed = False
        self._thread_state = _ThreadState()

    @property
    def closed(self) -> bool:
        """Whether the executor has been closed.

        Returns:
            `True` once the executor was closed.
        """
        with self._lock:
            return self._closed

    def run(
        self, calls: Sequence[WorkItem], *, bundle_size: int | None = None
    ) -> list[Any]:
        """Run the calls and return their results, in the order they were given.

        Args:
            calls:       The work items to run.
            bundle_size: Calls per worker task, `0` for all of them.

        Returns:
            One result per call, in the order of `calls`.

        Raises:
            WorkflowError: If the executor is closed, or the caller is one of
                           its workers.
        """
        if self._thread_state.running_work_item:
            raise WorkflowError(_ON_WORKER)
        with self._lock:
            if self._closed:
                raise WorkflowError(_CLOSED)
        items = list(calls)
        if not items:
            return []
        size = self._resolve_bundle_size(len(items), bundle_size)
        bundles = [items[start : start + size] for start in range(0, len(items), size)]
        results: list[Any] = [None] * len(items)

        def store(index: int, bundle_result: Any) -> None:  # ruff: ignore[any-type]
            _store(results, index * size, len(bundles[index]), bundle_result)

        self._run_bundles(bundles, store)
        return results

    def close(self) -> None:
        """Release the executor's workers.

        May be called from any thread, and more than once.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._on_close()
        self._release()

    def _resolve_bundle_size(self, count: int, bundle_size: int | None) -> int:
        if bundle_size is None:
            size = self._default_bundle_size
        else:
            _check_bundle_size(bundle_size)
            size = bundle_size
        # A bundle never spans batches, so zero is the whole batch.
        return count if size == 0 else size

    def _on_close(self) -> None:
        """Note the close while the lock is held."""

    @abstractmethod
    def _release(self) -> None:
        """Release the executor's resources, with the lock not held."""

    @abstractmethod
    def _run_bundles(
        self,
        bundles: list[list[WorkItem]],
        store: Callable[[int, Any], None],
    ) -> None:
        """Run the bundles, passing each one's result to `store` as it arrives.

        `store` takes the index of the bundle in `bundles` and either the list
        of its results or an
        [`ExecutorFailure`][ropt.components.executors.ExecutorFailure] for the
        whole bundle. It may raise, which is how a user-code exception reaches
        the caller: whatever this batch started must be released before that
        exception leaves.

        Args:
            bundles: The bundles to run.
            store:   Callback taking a bundle index and that bundle's result.
        """


def _check_bundle_size(bundle_size: int) -> None:
    if bundle_size < 0:
        msg = f"bundle_size must be >= 0, got {bundle_size}"
        raise ValueError(msg)


def _store(results: list[Any], offset: int, count: int, bundle_result: Any) -> None:  # ruff: ignore[any-type]
    if not isinstance(bundle_result, ExecutorFailure) and (
        not isinstance(bundle_result, list) or len(bundle_result) != count
    ):
        bundle_result = ExecutorFailure(
            f"A job returned a result that does not match its {count} work item(s)."
        )
    if isinstance(bundle_result, ExecutorFailure):
        results[offset : offset + count] = [bundle_result] * count
        return
    results[offset : offset + count] = bundle_result


def _stopped() -> ExecutorStopped:
    return ExecutorStopped(_STOPPED)
