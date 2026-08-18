"""This module implements the default multiprocessing executor."""

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


class MultiprocessingExecutor(ExecutorBase):
    """An executor that employs a pool of multiprocessing workers.

    See [Parallel Evaluation](../workflows/parallel.md#multiprocessingexecutor) for
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
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=self._max_tasks_per_child,
        )
        self._executor = executor
        _logger.debug(
            "Starting multiprocessing executor with %d worker(s)", self._workers
        )
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
            self._executor.shutdown(wait=False, cancel_futures=True)
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
                _logger.warning("Worker process pool broken; work item result lost")
                self._deliver(
                    submission,
                    work_item,
                    ExecutorFailure("Background process was killed"),
                )
            except asyncio.CancelledError:
                self._abort(submission)
                raise
            except BaseException as exc:
                self._fail(submission, exc)
                if not isinstance(exc, Exception):
                    raise


async def _run_work_item(work_item: WorkItem, executor: ProcessPoolExecutor) -> Any:  # ruff: ignore[any-type]
    loop = asyncio.get_running_loop()
    if _HAVE_CLOUDPICKLE:
        payload = cloudpickle.dumps(
            (work_item.function, work_item.args, work_item.kwargs)
        )
        ok, blob = await loop.run_in_executor(executor, _run_cloudpickled, payload)
        value = cloudpickle.loads(blob)
        if not ok:
            raise value
        return value
    try:
        pickle.dumps(work_item.function)
    except Exception as exc:
        msg = (
            "The work item function could not be sent to a worker process "
            "because it is not picklable; install ropt[cloudpickle] or make it "
            "picklable."
        )
        raise ExecutionError(msg) from exc
    return await loop.run_in_executor(
        executor, _run_function, work_item.function, work_item.args, work_item.kwargs
    )


def _run_function(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:  # ruff: ignore[any-type]
    check_transferred()
    return function(*args, **kwargs)


def _run_cloudpickled(payload: bytes) -> tuple[bool, bytes]:
    reset_transferred()
    try:
        function, args, kwargs = cloudpickle.loads(payload)
        check_transferred()
        return True, cloudpickle.dumps(function(*args, **kwargs))
    except Exception as exc:  # ruff: ignore[blind-except]
        return False, cloudpickle.dumps(picklable_exception(exc))


def _canary() -> None:
    pass
