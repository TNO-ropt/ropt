"""This module implements the process-based executor."""

from __future__ import annotations

import contextlib
import multiprocessing
import queue
import signal
import threading
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, override

from ropt._components.concurrency import AbortScope, _grace
from ropt._logging import get_logger
from ropt._serialize import CANNOT_DESERIALIZE, CANNOT_SERIALIZE, dumps, loads
from ropt.enums import ExitCode
from ropt.exceptions import ExecutionError

from ._picklable import picklable_exception
from .base import (
    _NOT_RUN,
    ExecutorBase,
    ExecutorFailure,
    WorkItem,
    WorkNotRun,
    _log_left_running,
    _run_item,
    _stopped,
    _wait_for,
)

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable

_logger = get_logger(__name__)

_POOL_GONE = "The work item was not sent: the worker processes are gone."

# How often a worker looks for an abort of the item it runs.
_FLAG_INTERVAL = 0.05

type _Flags = ctypes.Array[ctypes.c_int]
type _Result = Future[tuple[bool, bytes]]


class ProcessExecutor(ExecutorBase):
    """An executor that employs a pool of multiprocessing workers.

    See [Evaluating in worker processes](../running/parallel.md#process-pool)
    for details, and [Your evaluation function](../troubleshooting/index.md#your-evaluation-function)
    for the `if __name__ == "__main__":` guard that the entry point must use.

    An abort reaches a work item that is still running: code in it that
    calls [`aborting`][ropt.aborting], or runs a program with
    [`run_program`][ropt.run_program], sees it. When this executor belongs to a
    [`Session`][ropt.Session], an abort of the session also kills the worker
    processes still busy five seconds later.

    Note:
        An abort also leaves one item running that was submitted but not yet
        started: the pool marks an item as running when it moves it towards a
        worker, so it can no longer be withdrawn. Such an item sees the abort
        when it starts, and returns at once. See
        [Stopping a run](../running/parallel.md#stopping-a-run).
    """

    def __init__(
        self,
        *,
        workers: int = 1,
        max_tasks_per_child: int | None = None,
    ) -> None:
        """Initialize the executor.

        The worker processes are started here, so an entry point that is not
        guarded with `if __name__ == "__main__":` fails at this call.

        Args:
            workers:             Number of worker processes.
            max_tasks_per_child: Restart workers after this many items, or never.

        Raises:
            ValueError:     If `workers` is less than one.
            ExecutionError: If the worker processes could not be started.
        """  # ruff: ignore[docstring-extraneous-exception]
        super().__init__()
        if workers < 1:
            msg = f"The number of workers must be at least one: {workers}"
            raise ValueError(msg)
        self._workers = workers
        self._max_tasks_per_child = max_tasks_per_child
        self._context = multiprocessing.get_context("spawn")
        # A submitted item's pickled bytes stay in memory until its result
        # comes back, so this caps how many are submitted at once: one for each
        # worker, plus one so a worker that finishes finds the next item
        # already waiting. Each slot also carries the abort flag of its item.
        slots = workers + 1
        self._flags: _Flags = self._context.RawArray("i", slots)
        self._payload_gate = _PayloadGate(self._flags, slots)
        # Guards which worker set is current. Never taken by a future's done
        # callback: the pool runs those holding a lock of its own, which
        # `submit` and `shutdown` take while this one is held.
        self._state = threading.Lock()
        self._retiring = False
        self._current: _Workers | None = self._start_workers()
        _logger.debug("Started process executor with %d worker(s)", workers)

    def _start_workers(self) -> _Workers:
        pool = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=self._context,
            max_tasks_per_child=self._max_tasks_per_child,
            initializer=_init_worker,
            initargs=(self._flags, _grace.GRACE_PERIOD),
        )
        try:
            pool.submit(_dummy).result()
        except BrokenProcessPool as exc:
            _stop_workers(pool, kill=False)
            msg = (
                "Could not start worker processes; guard the program entry point "
                'with `if __name__ == "__main__":`.'
            )
            raise ExecutionError(msg) from exc
        return _Workers(pool)

    def _live_workers(self) -> _Workers:
        # Called with `_state` held. `_current` is `None` only once closed.
        if self._current is None:
            raise _stopped()
        return self._current

    @override
    def _retire_after_abort(self) -> None:
        with self._state:
            workers = self._current
            if self._retiring or workers is None or not workers.busy():
                return
            self._retiring = True
        deadline = time.monotonic() + _grace.GRACE_PERIOD
        threading.Thread(
            target=self._kill_when_due, args=(workers, deadline), daemon=True
        ).start()

    def _kill_when_due(self, workers: _Workers, deadline: float) -> None:
        workers.wait_idle(deadline)
        with self._state:
            if self._current is workers and workers.busy():
                _stop_workers(workers.pool, kill=True)

    @override
    def _finish_shutdown(self, deadline: float) -> None:
        with self._state:
            workers, self._current = self._current, None
        if workers is None:
            return
        try:
            idle = workers.wait_idle(deadline)
        except BaseException:
            _stop_workers(workers.pool, kill=True)
            raise
        if idle:
            workers.pool.shutdown(wait=False, cancel_futures=True)
        else:
            _stop_workers(workers.pool, kill=True)

    @override
    def _run_items(
        self,
        items: list[WorkItem],
        store: Callable[[int, Any], None],
        abort_scope: AbortScope | None,
    ) -> None:
        pending = deque(enumerate(items))
        # Finished items arrive here rather than through
        # `concurrent.futures.wait`, which would block forever on a future
        # cancelled before the pool dispatched it; a done callback still fires.
        done: queue.SimpleQueue[_Result | None] = queue.SimpleQueue()
        futures: dict[_Result, tuple[int, _Slot]] = {}
        # The waits this call can be parked on are the result queue and the
        # payload gate; an abort has to break both.
        wake = partial(done.put, None)
        if abort_scope is not None:
            abort_scope.add_callback(wake)
            abort_scope.add_callback(self._payload_gate.wake)
        try:
            while pending or futures:
                if abort_scope is not None and abort_scope.aborting:
                    break
                unsent = self._submit_pending(pending, futures, done, abort_scope)
                _store_unsent(unsent, store)
                if not futures:
                    continue
                item = done.get()
                if item is None:
                    break
                index, slot = futures.pop(item)
                self._payload_gate.release(slot)
                if not item.cancelled():
                    store(index, _item_result(item))
        finally:
            if abort_scope is not None:
                abort_scope.remove_callback(wake)
                abort_scope.remove_callback(self._payload_gate.wake)
            for future, (_, slot) in futures.items():
                if future.cancel():
                    self._payload_gate.release(slot)
                else:
                    self._leave_running(future, slot, abort_scope)

    def _leave_running(
        self, future: _Result, slot: _Slot, abort_scope: AbortScope | None
    ) -> None:
        # The item keeps its parent: an abort of the scope that sent it, now
        # or later, still reaches it, as it would on a thread.
        forward = None
        if abort_scope is not None:
            forward = partial(self._payload_gate.abort, slot, abort_scope)
            abort_scope.add_callback(forward)
        future.add_done_callback(
            partial(self._end_left_running, slot, abort_scope, forward)
        )

    def _end_left_running(
        self,
        slot: _Slot,
        abort_scope: AbortScope | None,
        forward: Callable[[], None] | None,
        future: _Result,
    ) -> None:
        if abort_scope is not None and forward is not None:
            abort_scope.remove_callback(forward)
        # A payload holds its slot until its work ends, waited for or not.
        self._payload_gate.release(slot)
        if future.cancelled() or isinstance(future.exception(), BrokenProcessPool):
            # Killed by an abort, or with a pool that is reported broken anyway.
            return
        _log_left_running(_item_result, future)

    def _submit_pending(
        self,
        pending: deque[tuple[int, WorkItem]],
        futures: dict[_Result, tuple[int, _Slot]],
        done: queue.SimpleQueue[_Result | None],
        abort_scope: AbortScope | None,
    ) -> list[int]:
        # Waiting for a slot is safe only with nothing in flight: the slots this
        # batch holds are given back by the loop that calls this.
        while pending:
            slot = self._payload_gate.acquire(
                blocking=not futures, abort_scope=abort_scope
            )
            if slot is None:
                break
            index, item = pending.popleft()
            try:
                future = self._submit(item, slot)
            except BrokenProcessPool:
                self._payload_gate.release(slot)
                # A broken pool is not replaced, so the rest of the batch is
                # lost too and building its payloads would run nothing. Killed
                # by an abort, the items are left unstored: they were not run.
                unsent = [index, *(other for other, _ in pending)]
                pending.clear()
                if abort_scope is not None and abort_scope.aborting:
                    return []
                return unsent
            except BaseException:
                self._payload_gate.release(slot)
                raise
            futures[future] = (index, slot)
            future.add_done_callback(done.put)
        return []

    def _submit(self, item: WorkItem, slot: _Slot) -> _Result:
        # The payload is built here rather than in a worker, so that `ropt`'s
        # own serialization is used and a failure to build it is raised in the
        # caller instead of surfacing as an opaque pool failure.
        try:
            payload = dumps(
                (_run_item, (item.function, item.args, item.kwargs), {"sendable": True})
            )
        except Exception as exc:
            msg = (
                "The work item could not be sent to a worker process: "
                f"{CANNOT_SERIALIZE}."
            )
            raise ExecutionError(msg) from exc
        try:
            with self._state:
                workers = self._live_workers()
                return workers.track(
                    workers.pool.submit(_run_payload, payload, slot.index)
                )
        except BrokenProcessPool:
            # Caught ahead of RuntimeError, which it subclasses, so that a dead
            # pool is not mistaken for interpreter shutdown below.
            raise
        except RuntimeError:
            # The pool is gone, which at interpreter shutdown is how a caller
            # that outlived its program is released rather than left waiting.
            raise _stopped() from None


def _store_unsent(unsent: list[int], store: Callable[[int, Any], None]) -> None:
    for index in unsent:
        store(index, ExecutorFailure(_POOL_GONE))


def _item_result(future: _Result) -> Any:  # ruff: ignore[any-type]
    try:
        ok, blob = future.result()
    except BrokenProcessPool:
        _logger.warning("Worker process pool broken; work item result lost")
        return ExecutorFailure("Background process was killed")
    value = loads(blob)
    if not ok:
        raise value
    return value


def _stop_workers(executor: ProcessPoolExecutor, *, kill: bool) -> None:
    stop_workers = getattr(
        executor, "kill_workers" if kill else "terminate_workers", None
    )
    if stop_workers is not None:
        # Python 3.14 and later. This shuts the pool down as part of its job.
        stop_workers()
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
        with contextlib.suppress(ValueError, ProcessLookupError):
            if process.is_alive():
                stop = process.kill if kill else process.terminate
                stop()


class _WorkerFlags:
    # Set in each worker process by `_init_worker`.
    flags: _Flags | None = None


def _init_worker(flags: _Flags, grace: float) -> None:
    # Ctrl-C is the main process's to handle: a worker that raised it would
    # report the interrupt as the result of whatever it was running.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WorkerFlags.flags = flags
    _grace.GRACE_PERIOD = grace


def _watch_flag(
    flags: _Flags, index: int, abort_scope: AbortScope, finished: threading.Event
) -> None:
    while not finished.wait(_FLAG_INTERVAL):
        code = flags[index]
        if code:
            abort_scope.abort(ExitCode(code))
            return


def _run_payload(payload: bytes, index: int) -> tuple[bool, bytes]:
    flags = _WorkerFlags.flags
    if flags is None:
        return _run_unpacked(payload)
    if flags[index]:
        return True, dumps(WorkNotRun(_NOT_RUN))
    abort_scope = AbortScope()
    finished = threading.Event()
    threading.Thread(
        target=_watch_flag, args=(flags, index, abort_scope, finished), daemon=True
    ).start()
    try:
        with abort_scope.as_current():
            return _run_unpacked(payload)
    finally:
        finished.set()


def _run_unpacked(payload: bytes) -> tuple[bool, bytes]:
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


class _Workers:
    # One set of worker processes, and the items submitted to it that have not
    # finished. The condition is the one lock a done callback takes here.

    def __init__(self, pool: ProcessPoolExecutor) -> None:
        self.pool = pool
        self._changed = threading.Condition()
        self._running: set[_Result] = set()

    def track(self, future: _Result) -> _Result:
        with self._changed:
            self._running.add(future)
        future.add_done_callback(self._finished)
        return future

    def busy(self) -> bool:
        with self._changed:
            return bool(self._running)

    def wait_idle(self, deadline: float) -> bool:
        return _wait_for(self._changed, lambda: not self._running, deadline)

    def _finished(self, future: _Result) -> None:
        with self._changed:
            self._running.discard(future)
            self._changed.notify_all()


@dataclass(eq=False)
class _Slot:
    # Identifies one use of a slot, so that a late abort of an item that has
    # finished does not reach the item that reused its slot.
    index: int


class _PayloadGate:
    # The slots are shared by every concurrent caller of one executor, so a
    # caller with nothing in flight waits here for work that is not its own. A
    # semaphore has no way back out of that wait, hence the condition.

    def __init__(self, flags: _Flags, slots: int) -> None:
        self._condition = threading.Condition()
        self._flags = flags
        self._free = list(range(slots))
        self._owners: list[_Slot | None] = [None] * slots

    def acquire(
        self, *, blocking: bool, abort_scope: AbortScope | None = None
    ) -> _Slot | None:
        with self._condition:
            while True:
                # Tested before the slot, so a waiter woken by a freed slot
                # rather than by the abort does not take it.
                if abort_scope is not None and abort_scope.aborting:
                    return None
                if self._free:
                    slot = _Slot(self._free.pop())
                    self._owners[slot.index] = slot
                    self._flags[slot.index] = 0
                    return slot
                if not blocking:
                    return None
                self._condition.wait()

    def release(self, slot: _Slot) -> None:
        with self._condition:
            if self._owners[slot.index] is not slot:
                return
            self._owners[slot.index] = None
            self._free.append(slot.index)
            self._condition.notify()

    def abort(self, slot: _Slot, abort_scope: AbortScope) -> None:
        with self._condition:
            if self._owners[slot.index] is slot:
                self._flags[slot.index] = int(abort_scope.exit_code)

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()
