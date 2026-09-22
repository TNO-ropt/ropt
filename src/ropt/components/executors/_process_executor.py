"""This module implements the process-based executor."""

from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, cast

from ropt._logging import get_logger
from ropt._serialize import CANNOT_DESERIALIZE, CANNOT_SERIALIZE, dumps, loads
from ropt.exceptions import ExecutionError

from ._picklable import picklable_exception
from .base import ExecutorBase, ExecutorFailure, WorkItem, _calls, _run_bundle

_logger = get_logger(__name__)


class ProcessExecutor(ExecutorBase):
    """An executor that employs a pool of multiprocessing workers.

    See [Parallel Evaluation](../advanced/parallel.md#processexecutor) for
    details, including the `if __name__ == "__main__":` guard that the entry
    point must use.

    Warning:
        Stopping terminates the worker processes and nothing else. A program a
        work item started itself keeps running, without an error being raised.
        Use [`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor]
        where an evaluation launches external programs.
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
            max_tasks_per_child: Restart workers after this many items, or never.
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
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, _dummy)
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
            # Terminate the workers, or we have to wait for them.
            _terminate_workers(self._executor)
            self._executor = None
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        self._cleanup_submissions()

    async def _run_worker(self, executor: ProcessPoolExecutor) -> None:
        while True:
            submission, bundle = await self._work_queue.get()
            if submission.is_finished:
                continue
            try:
                results = await _run_bundle_in_pool(bundle, executor)
                for work_item, result in zip(bundle, results, strict=True):
                    self._deliver(submission, work_item, result)
            except BrokenProcessPool:
                if self._running.is_set():
                    _logger.warning("Worker process pool broken; work item result lost")
                else:
                    _logger.debug("Work item dropped: the executor was stopped")
                for work_item in bundle:
                    self._deliver(
                        submission,
                        work_item,
                        ExecutorFailure("Background process was killed"),
                    )
            except asyncio.CancelledError:
                self._abort(submission)
                raise
            except BaseException as exc:
                #  Reraise `Exception` (SystemExit, KeyboardInterrupt).
                self._fail(submission, exc)
                if not isinstance(exc, Exception):
                    raise


def _terminate_workers(executor: ProcessPoolExecutor) -> None:
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
            process.terminate()
        except (ValueError, ProcessLookupError):
            continue


async def _run_bundle_in_pool(
    bundle: list[WorkItem], executor: ProcessPoolExecutor
) -> list[Any]:
    loop = asyncio.get_running_loop()
    try:
        payload = dumps((_run_bundle, (_calls(bundle),), {}))
    except Exception as exc:
        msg = (
            f"The work item could not be sent to a worker process: {CANNOT_SERIALIZE}."
        )
        raise ExecutionError(msg) from exc
    ok, blob = await loop.run_in_executor(executor, _run_payload, payload)
    value = loads(blob)
    if not ok:
        raise value
    return cast("list[Any]", value)


def _run_payload(payload: bytes) -> tuple[bool, bytes]:
    try:
        function, args, kwargs = loads(payload)
    except Exception as exc:  # ruff: ignore[blind-except]
        exc.add_note(f"Could not rebuild the work item: {CANNOT_DESERIALIZE}.")
        return False, dumps(picklable_exception(exc))
    try:
        value = function(*args, **kwargs)
    except Exception as exc:  # ruff: ignore[blind-except]
        # Return exception rather than raising it, so it can be sent back to the caller.
        return False, dumps(picklable_exception(exc))
    try:
        return True, dumps(value)
    except Exception as exc:  # ruff: ignore[blind-except]
        exc.add_note(f"Could not send the result back: {CANNOT_SERIALIZE}.")
        return False, dumps(picklable_exception(exc))


def _dummy() -> None:
    pass
