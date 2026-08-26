"""This module implements the process-based executor."""

from __future__ import annotations

import asyncio
import multiprocessing
import pickle  # ruff: ignore[suspicious-pickle-import]
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final

from ropt._logging import get_logger
from ropt.components._transferred import check_transferred, reset_transferred
from ropt.exceptions import ExecutionError, ExecutorFailure

from .base import ExecutorBase, WorkItem

if TYPE_CHECKING:
    from collections.abc import Callable

_HAVE_CLOUDPICKLE: Final = find_spec("cloudpickle") is not None

if _HAVE_CLOUDPICKLE:
    import cloudpickle

    from ._picklable import picklable_exception

_logger = get_logger(__name__)


class ProcessExecutor(ExecutorBase):
    """An executor that employs a pool of multiprocessing workers.

    See [Parallel Evaluation](../workflows/parallel.md#processexecutor) for
    details, including the `if __name__ == "__main__":` guard that the entry
    point must use.
    """

    def __init__(
        self,
        *,
        workers: int = 1,
        max_tasks_per_child: int | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            workers:             Number of worker processes.
            max_tasks_per_child: Restart workers after this many work items
                                 (`None` = never).
        """
        super().__init__()
        self._workers = workers
        self._max_tasks_per_child = max_tasks_per_child
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._executor: ProcessPoolExecutor | None = None

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group:          The task group to use.
        """
        self._begin_start()
        executor = ProcessPoolExecutor(
            max_workers=self._workers,
            # Spawn rather than fork: the parent runs an event loop and threads,
            # which a forked child does not inherit in a usable state.
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=self._max_tasks_per_child,
        )
        self._executor = executor
        _logger.debug("Starting process executor with %d worker(s)", self._workers)
        await self._check_worker_startup()
        self._worker_tasks = [
            task_group.create_task(self._run_worker(executor))
            for _ in range(self._workers)
        ]
        await self._finish_start(task_group)

    async def _check_worker_startup(self) -> None:
        # A missing `__main__` guard only shows up when a worker actually
        # starts, so a trivial task is run first: the failure then names its
        # cause, instead of surfacing on whichever work item came first.
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, _canary)
        except BrokenProcessPool as exc:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            msg = (
                "Could not start worker processes; guard the program entry point "
                'with `if __name__ == "__main__":`.'
            )
            raise ExecutionError(msg) from exc

    def _cleanup(self) -> None:
        """Clean up the executor."""
        if self._executor is not None:
            # Stopping means stopping: without this the pool waits for whatever
            # the workers are busy with, which is the whole reason an
            # interrupted program appears to hang.
            _terminate_workers(self._executor)
            self._executor = None
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        self._cleanup_submissions()

    async def _run_worker(self, executor: ProcessPoolExecutor) -> None:
        while True:
            submission, work_item = await self._work_queue.get()
            if submission.is_finished:
                # Its caller has already left, so running this wastes a worker.
                continue
            try:
                result = await _run_work_item(work_item, executor)
                self._deliver(submission, work_item, result)
            except BrokenProcessPool:
                if self._running.is_set():
                    # A killed worker is infrastructure, not user code: it is
                    # delivered as a failed result and the pool keeps going,
                    # since the process pool replaces the worker on its own.
                    _logger.warning("Worker process pool broken; work item result lost")
                else:
                    # A stop usually ends in `CancelledError` instead, but the
                    # two race: `task.cancel()` only schedules the error, while
                    # the killed future may already have its exception set.
                    # `_wait_for_cancel` clears `_running` before `_cleanup`
                    # kills the workers, so whenever this arm does win, the
                    # dead worker is one we killed on purpose rather than an
                    # infrastructure failure worth a warning per busy worker.
                    _logger.debug("Work item dropped: the executor was stopped")
                self._deliver(
                    submission,
                    work_item,
                    ExecutorFailure("Background process was killed"),
                )
            except asyncio.CancelledError:
                self._abort(submission)
                raise
            except BaseException as exc:
                # The caller is told either way, but an exception that is not an
                # `Exception` (SystemExit, KeyboardInterrupt) is not this run's
                # to swallow: it keeps unwinding into the task group.
                self._fail(submission, exc)
                if not isinstance(exc, Exception):
                    raise


def _terminate_workers(executor: ProcessPoolExecutor) -> None:
    # Deliberately narrow: no process groups, no grandchildren, no output
    # capture. This stops the pool's own workers, nothing else, and the only
    # reason it is a function is that how to do it depends on the version.
    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:
        # Python 3.14 and later. This shuts the pool down as part of its job.
        terminate_workers()
        return

    # The same algorithm by hand. The lock is taken for one thing only: reading
    # `_processes` without racing a concurrent mutation. It must be released
    # before `shutdown`, which acquires the same non-reentrant lock itself.
    with executor._shutdown_lock:  # ruff: ignore[private-member-access]
        processes = list((executor._processes or {}).values())  # ruff: ignore[private-member-access]

    # Never wait: a worker that decides not to exit would deadlock the caller,
    # which is what CPython refuses to risk here too. `shutdown` invalidates
    # `_processes`, hence the copy above.
    executor.shutdown(wait=False, cancel_futures=True)

    # A worker started between the snapshot and here is not signalled. That gap
    # is CPython's gh-152967 and is not guarded on these versions: closing it
    # needs the internal "force shutting down" flag that only 3.14 and later
    # have, so the alternative would be a wait, which is what this removes.
    for process in processes:
        try:
            if not process.is_alive():
                continue
            # SIGTERM only. Escalating to SIGKILL after a timeout would put a
            # wait back on the stopping path; a process that ignores SIGTERM or
            # sits in uninterruptible sleep is not helped by it anyway.
            process.terminate()
        except (ValueError, ProcessLookupError):
            # Already exited, closed out, or gone between the two calls.
            continue


async def _run_work_item(work_item: WorkItem, executor: ProcessPoolExecutor) -> Any:  # ruff: ignore[any-type]
    loop = asyncio.get_running_loop()
    if _HAVE_CLOUDPICKLE:
        # Serialized here rather than by the pool, so that closures, lambdas and
        # locally defined functions can be sent as well.
        payload = cloudpickle.dumps(
            (work_item.function, work_item.args, work_item.kwargs)
        )
        ok, blob = await loop.run_in_executor(executor, _run_cloudpickled, payload)
        value = cloudpickle.loads(blob)
        if not ok:
            raise value
        return value
    # Without cloudpickle the pool does the pickling, and reports a failure from
    # inside its own machinery. Checking first turns that into a clear message.
    try:
        pickle.dumps((work_item.function, work_item.args, work_item.kwargs))
    except Exception as exc:
        msg = (
            "The work item could not be sent to a worker process because its "
            "function or arguments are not picklable; install "
            "ropt[cloudpickle] or make them picklable."
        )
        raise ExecutionError(msg) from exc
    return await loop.run_in_executor(
        executor, _run_function, work_item.function, work_item.args, work_item.kwargs
    )


def _run_function(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:  # ruff: ignore[any-type]
    # The pool unpickled the arguments before this call, so any workflow object
    # among them has already been recorded.
    check_transferred()
    return function(*args, **kwargs)


def _run_cloudpickled(payload: bytes) -> tuple[bool, bytes]:
    reset_transferred()
    try:
        function, args, kwargs = cloudpickle.loads(payload)
        check_transferred()
        return True, cloudpickle.dumps(function(*args, **kwargs))
    except Exception as exc:  # ruff: ignore[blind-except]
        # Returned rather than raised: the pool sends an exception back with the
        # standard pickle module, which not every exception survives.
        return False, cloudpickle.dumps(picklable_exception(exc))


# Trivial by design: it proves a worker can start, nothing more.
def _canary() -> None:
    pass
