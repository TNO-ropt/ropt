"""This module implements the process-based executor."""

from __future__ import annotations

import multiprocessing
import queue
import threading
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
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

    from ropt.components.concurrency import AbortSignal

_logger = get_logger(__name__)


class ProcessExecutor(ExecutorBase):
    """An executor that employs a pool of multiprocessing workers.

    See [Parallel Evaluation](../advanced/parallel.md#processexecutor) for
    details, including the `if __name__ == "__main__":` guard that the entry
    point must use.

    Warning:
        A worker process runs its work item to the end. A program a work item
        started itself keeps running, without an error being raised. Use
        [`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] where
        an evaluation launches external programs.

    Note:
        An abort also runs one bundle that was submitted but not yet started:
        the pool marks a bundle as running when it moves it towards a worker,
        so it can no longer be withdrawn. See
        [Releasing a batch](../advanced/parallel.md#releasing-a-batch).
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
        self._payload_gate = _PayloadGate(workers + 1)
        self._check_worker_startup()
        _logger.debug("Started process executor with %d worker(s)", workers)

    def _check_worker_startup(self) -> None:
        try:
            self._pool.submit(_dummy).result()
        except BrokenProcessPool as exc:
            _terminate_workers(self._pool)
            msg = (
                "Could not start worker processes; guard the program entry point "
                'with `if __name__ == "__main__":`.'
            )
            raise ExecutionError(msg) from exc

    def _run_bundles(
        self,
        bundles: list[list[WorkItem]],
        store: Callable[[int, Any], None],
        abort_signal: AbortSignal | None,
    ) -> None:
        pending = deque(enumerate(bundles))
        # Finished bundles arrive here rather than through
        # `concurrent.futures.wait`, which would block forever on a future
        # cancelled before the pool dispatched it; a done callback still fires.
        done: queue.SimpleQueue[Future[tuple[bool, bytes]] | None] = queue.SimpleQueue()
        futures: dict[Future[tuple[bool, bytes]], int] = {}
        # The waits this call can be parked on are the result queue and the
        # payload gate; an abort has to break both.
        wake = partial(done.put, None)
        if abort_signal is not None:
            abort_signal.add_callback(wake)
            abort_signal.add_callback(self._payload_gate.wake)
        try:
            while pending or futures:
                if abort_signal is not None and abort_signal.aborting:
                    # Work not yet sent is dropped here; work already with a
                    # worker is collected below, since it runs whether or not
                    # anyone is still waiting for it.
                    pending.clear()
                self._submit_pending(pending, futures, done, abort_signal)
                if not futures:
                    continue
                item = done.get()
                if item is None:
                    for future in list(futures):
                        future.cancel()
                    continue
                index = futures.pop(item)
                self._payload_gate.release()
                if not item.cancelled():
                    store(index, _bundle_result(item))
        finally:
            if abort_signal is not None:
                abort_signal.remove_callback(wake)
                abort_signal.remove_callback(self._payload_gate.wake)
            for future in futures:
                future.cancel()
                self._payload_gate.release()

    def _submit_pending(
        self,
        pending: deque[tuple[int, list[WorkItem]]],
        futures: dict[Future[tuple[bool, bytes]], int],
        done: queue.SimpleQueue[Future[tuple[bool, bytes]] | None],
        abort_signal: AbortSignal | None,
    ) -> None:
        # Waiting for a slot is safe only with nothing in flight: the slots this
        # batch holds are given back by the loop that calls this.
        while pending and self._payload_gate.acquire(
            blocking=not futures, abort_signal=abort_signal
        ):
            index, bundle = pending.popleft()
            try:
                future = self._submit(bundle)
            except BaseException:
                self._payload_gate.release()
                raise
            futures[future] = index
            future.add_done_callback(done.put)

    def _submit(self, bundle: list[WorkItem]) -> Future[tuple[bool, bytes]]:
        # The payload is built here rather than in a worker, so that `ropt`'s
        # own serialization is used and a failure to build it is raised in the
        # caller instead of surfacing as an opaque pool failure.
        try:
            payload = dumps((_run_bundle, (_calls(bundle),), {"sendable": True}))
        except Exception as exc:
            msg = (
                "The work item could not be sent to a worker process: "
                f"{CANNOT_SERIALIZE}."
            )
            raise ExecutionError(msg) from exc
        try:
            return self._pool.submit(_run_payload, payload)
        except BrokenProcessPool as exc:
            # Checked ahead of RuntimeError, which it subclasses: every worker
            # is gone and the pool cannot be restarted, only replaced.
            msg = "The worker processes are gone; this executor cannot run more work."
            raise ExecutionError(msg) from exc
        except RuntimeError:
            # The pool is gone, which at interpreter shutdown is how a caller
            # that outlived its program is released rather than left waiting.
            raise _stopped() from None


def _bundle_result(
    future: Future[tuple[bool, bytes]],
) -> list[Any] | ExecutorFailure:
    try:
        ok, blob = future.result()
    except BrokenProcessPool:
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


class _PayloadGate:
    # The slots are shared by every concurrent caller of one executor, so a
    # caller with nothing in flight waits here for work that is not its own. A
    # semaphore has no way back out of that wait, hence the condition.

    def __init__(self, slots: int) -> None:
        self._condition = threading.Condition()
        self._free = slots

    def acquire(
        self, *, blocking: bool, abort_signal: AbortSignal | None = None
    ) -> bool:
        with self._condition:
            while self._free == 0:
                if not blocking or (abort_signal is not None and abort_signal.aborting):
                    return False
                self._condition.wait()
            self._free -= 1
            return True

    def release(self) -> None:
        with self._condition:
            self._free += 1
            self._condition.notify()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()
