"""This module implements the process-based executor."""

from __future__ import annotations

import multiprocessing
import queue
import threading
from collections import deque
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any, cast

from ropt._logging import get_logger
from ropt._serialize import CANNOT_DESERIALIZE, CANNOT_SERIALIZE, dumps, loads
from ropt.exceptions import ExecutionError

from ._picklable import picklable_exception
from .base import (
    ExecutorBase,
    ExecutorFailure,
    WorkItem,
    _calls,
    _run_bundle,
    _stopped,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = get_logger(__name__)


class ProcessExecutor(ExecutorBase):
    """An executor that employs a pool of multiprocessing workers.

    See [Parallel Evaluation](../advanced/parallel.md#processexecutor) for
    details, including the `if __name__ == "__main__":` guard that the entry
    point must use.

    Warning:
        Closing terminates the worker processes and nothing else. A program a
        work item started itself keeps running, without an error being raised.
        Use [`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor]
        where an evaluation launches external programs.
    """

    def __init__(
        self,
        *,
        workers: int = 1,
        max_tasks_per_child: int | None = None,
        bundle_size: int = 1,
    ) -> None:
        """Initialize the executor.

        The worker processes are started here, so an entry point that is not
        guarded with `if __name__ == "__main__":` fails at this call.

        Args:
            workers:             Number of worker processes.
            max_tasks_per_child: Restart workers after this many items, or never.
            bundle_size:         Calls per worker task, `0` for a whole batch.

        Raises:
            ValueError:     If `workers` is less than one.
            ExecutionError: If the worker processes could not be started.
        """  # ruff: ignore[docstring-extraneous-exception]
        super().__init__(bundle_size=bundle_size)
        if workers < 1:
            msg = f"The number of workers must be at least one: {workers}"
            raise ValueError(msg)
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=max_tasks_per_child,
        )
        # A submitted bundle's pickled bytes stay in memory until its result
        # comes back, so this caps how many are submitted at once: one for each
        # worker, plus one so a worker that finishes finds the next bundle
        # already waiting.
        self._payload_limit = threading.Semaphore(workers + 1)
        self._check_worker_startup()
        _logger.debug("Started process executor with %d worker(s)", workers)

    def _check_worker_startup(self) -> None:
        try:
            self._pool.submit(_dummy).result()
        except BrokenProcessPool as exc:
            self.close()
            msg = (
                "Could not start worker processes; guard the program entry point "
                'with `if __name__ == "__main__":`.'
            )
            raise ExecutionError(msg) from exc

    def _release(self) -> None:
        _terminate_workers(self._pool)

    def _run_bundles(
        self,
        bundles: list[list[WorkItem]],
        store: Callable[[int, Any], None],
    ) -> None:
        pending = deque(enumerate(bundles))
        # Finished bundles arrive here rather than through
        # `concurrent.futures.wait`, which would block forever on a future
        # cancelled before the pool dispatched it; a done callback still fires.
        done: queue.SimpleQueue[Future[tuple[bool, bytes]]] = queue.SimpleQueue()
        futures: dict[Future[tuple[bool, bytes]], int] = {}
        try:
            while pending or futures:
                # Waiting is safe only with nothing in flight: the slots this
                # batch holds are given back further down, by this same loop.
                while pending and self._payload_limit.acquire(blocking=not futures):
                    index, bundle = pending.popleft()
                    try:
                        future = self._submit(bundle)
                    except BaseException:
                        self._payload_limit.release()
                        raise
                    futures[future] = index
                    future.add_done_callback(done.put)
                future = done.get()
                index = futures.pop(future)
                self._payload_limit.release()
                store(index, self._bundle_result(future))
        finally:
            for future in futures:
                future.cancel()
                self._payload_limit.release()

    def _submit(self, bundle: list[WorkItem]) -> Future[tuple[bool, bytes]]:
        # The payload is built here rather than in a worker, so that `ropt`'s
        # own serialization is used and a failure to build it is raised in the
        # caller instead of surfacing as an opaque pool failure.
        try:
            payload = dumps((_run_bundle, (_calls(bundle),), {}))
        except Exception as exc:
            msg = (
                "The work item could not be sent to a worker process: "
                f"{CANNOT_SERIALIZE}."
            )
            raise ExecutionError(msg) from exc
        with self._lock:
            if self._closed:
                raise _stopped()
            try:
                return self._pool.submit(_run_payload, payload)
            except RuntimeError:
                raise _stopped() from None

    def _bundle_result(
        self, future: Future[tuple[bool, bytes]]
    ) -> list[Any] | ExecutorFailure:
        try:
            ok, blob = future.result()
        except CancelledError:
            raise _stopped() from None
        except BrokenProcessPool:
            if self.closed:
                # Closing is what killed the worker, so this is the stop the
                # caller asked for rather than infrastructure that broke.
                raise _stopped() from None
            _logger.warning("Worker process pool broken; work item result lost")
            return ExecutorFailure("Background process was killed")
        value = loads(blob)
        if not ok:
            raise value
        return cast("list[Any]", value)


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
