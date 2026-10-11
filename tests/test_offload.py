"""Tests for the offload method on sessions and worker pools."""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import threading
from functools import partial
from operator import add
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pytest

from ropt import session
from ropt.components.concurrency import AbortSignal, parent_signal
from ropt.components.event_handlers import EventHandler
from ropt.components.executors import ProcessExecutor, ThreadExecutor
from ropt.enums import EnOptEventType, ExitCode
from ropt.exceptions import AbortedError, ExecutionError, WorkflowError
from ropt.run._offload import _run

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ropt import WorkerPool
    from ropt.components.executors import Executor
    from ropt.events import EnOptEvent


@pytest.fixture(name="pools")
def pools_fixture() -> Iterator[Callable[..., WorkerPool]]:
    """Build pools on one session.

    Yields:
        A factory taking an executor class (`ThreadExecutor` by default) and its
        keyword arguments.
    """
    with session() as opened:
        factories: dict[type[Executor], Callable[..., WorkerPool]] = {
            ThreadExecutor: opened.thread_pool,
            ProcessExecutor: opened.process_pool,
        }

        def _make(kind: type[Executor] = ThreadExecutor, **kwargs: Any) -> WorkerPool:
            return factories[kind](**kwargs)

        yield _make


def _square(x: int) -> int:
    return x * x


def _double(x: int) -> int:
    return x + x


def _exit_process() -> int:
    sys.exit(3)


def _interrupt() -> int:
    raise KeyboardInterrupt


def _kill_worker() -> int:
    os._exit(1)


def test_offload_empty_sequence_returns_empty(
    pools: Callable[..., WorkerPool],
) -> None:
    assert pools(workers=1).offload([]) == ()


def test_offload_single_call_with_a_thread_pool(
    pools: Callable[..., WorkerPool],
) -> None:
    assert pools(workers=2).offload(partial(add, 3, 4)) == 7


class _ShutdownPool:
    # `concurrent.futures` refuses new work with `RuntimeError` once the
    # interpreter is shutting down.
    def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

    @staticmethod
    def submit(*_args: Any, **_kwargs: Any) -> None:
        msg = "cannot schedule new futures after interpreter shutdown"
        raise RuntimeError(msg)

    def shutdown(self, *_args: Any, **_kwargs: Any) -> None: ...


def test_offload_on_a_shut_down_pool_raises_aborted_with_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, pools: Callable[..., WorkerPool]
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._thread_executor.ThreadPoolExecutor",
        _ShutdownPool,
    )
    with pytest.raises(AbortedError) as raised:
        pools(workers=1).offload(partial(_square, 3))
    assert raised.value.exit_code is ExitCode.EXECUTOR_SHUT_DOWN


def test_offload_sequence_with_a_thread_pool(
    pools: Callable[..., WorkerPool],
) -> None:
    assert pools(workers=3).offload([partial(_square, i) for i in range(1, 6)]) == (
        1,
        4,
        9,
        16,
        25,
    )


def test_offload_sequence_of_different_functions(
    pools: Callable[..., WorkerPool],
) -> None:
    assert pools(workers=2).offload([partial(_square, 3), partial(_double, 5)]) == (
        9,
        10,
    )


@pytest.mark.slow
def test_offload_sequence_with_a_process_pool(
    pools: Callable[..., WorkerPool],
) -> None:
    assert pools(ProcessExecutor, workers=2).offload(
        [partial(_square, i) for i in (1, 2, 3)]
    ) == (
        1,
        4,
        9,
    )


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_dying_worker_reported_to_offload_caller(
    pools: Callable[..., WorkerPool],
) -> None:
    with pytest.raises(ExecutionError, match="could not be run"):
        pools(ProcessExecutor, workers=1).offload(_kill_worker)


def test_offload_preserves_order_across_workers(
    pools: Callable[..., WorkerPool],
) -> None:
    # Each job waits for its successor, so the jobs finish in exactly the
    # reverse of the order they were submitted in and a result tuple in
    # submission order can only come from reordering by index. One worker per
    # job, or the chain deadlocks on the pool.
    count = 5
    finished = [threading.Event() for _ in range(count)]

    def square(index: int) -> int:
        if index + 1 < count:
            assert finished[index + 1].wait(timeout=30)
        finished[index].set()
        return (index + 1) * (index + 1)

    jobs = [partial(square, index) for index in range(count)]
    assert pools(workers=count).offload(jobs) == (1, 4, 9, 16, 25)


def test_offload_from_an_event_loop(pools: Callable[..., WorkerPool]) -> None:
    # Nothing in the pools touches asyncio, so a call from inside a
    # coroutine -- a notebook cell, say -- is an ordinary blocking call.
    async def _offload_in_a_cell(pool: WorkerPool) -> int:  # ruff: ignore[unused-async]
        return pool.offload(partial(_square, 4))

    assert asyncio.run(_offload_in_a_cell(pools(workers=1))) == 16


@pytest.mark.timeout(30)
@pytest.mark.parametrize("work", [_exit_process, _interrupt])
def test_offload_base_exception_reaches_caller(
    pools: Callable[..., WorkerPool], work: Callable[[], int]
) -> None:
    # A worker thread cannot exit the interpreter on its own, so the exception
    # is delivered to the caller, which is where it means something.
    with pytest.raises((SystemExit, KeyboardInterrupt)):
        pools(workers=2).offload(work)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_offload_ended_by_a_base_exception_aborts_the_work_it_started(
    pools: Callable[..., WorkerPool], error: type[BaseException]
) -> None:
    # Work started from the offloaded function takes this signal as its parent,
    # so whether it aborts is what reaches that work.
    seen: list[AbortSignal] = []

    def _record_then_raise() -> int:
        signal = parent_signal()
        assert signal is not None
        seen.append(signal)
        raise error

    with pytest.raises(error):
        pools(workers=1).offload(_record_then_raise)
    assert seen[0].aborting
    assert seen[0].exit_code is ExitCode.ABORTED


class _OffloadingHandler(EventHandler):
    """Record what offloading from inside a handler returns."""

    def __init__(self, pool: WorkerPool) -> None:
        super().__init__()
        self.pool = pool
        self.outcome: int | None = None

    @property
    @override
    def event_types(self) -> set[EnOptEventType]:
        return {EnOptEventType.FINISHED_EVALUATION}

    @override
    def _handle_event(self, event: EnOptEvent) -> None:
        if self.outcome is None:
            self.outcome = self.pool.offload(partial(_square, 4))


@pytest.mark.timeout(60)
def test_handler_can_offload(pools: Callable[..., WorkerPool]) -> None:
    # A handler runs on the thread driving the run, not on a worker, so the
    # pool it is given works there as usual.
    pool = pools(workers=2)
    handler = _OffloadingHandler(pool)
    config = {
        "variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6},
        "optimizer": {"max_functions": 2},
    }
    pool.optimize(
        config,
        np.zeros(2),
        lambda variables, _: float(np.sum(variables**2)),
        handlers=[handler],
    )
    assert handler.outcome == 16


@pytest.mark.timeout(60)
def test_an_offload_aborted_while_its_function_runs_raises() -> None:
    # The function is already on a worker when the abort arrives. A worker
    # cannot be interrupted, and `offload` does not wait for it.
    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def _abort_then_return() -> int:
            opened.abort()
            return 7

        with pytest.raises(AbortedError) as raised:
            pool.offload(_abort_then_return)
    assert raised.value.exit_code is ExitCode.USER_ABORT


@pytest.mark.parametrize("exit_code", [ExitCode.ABORTED, ExitCode.ABORTED_ON_ERROR])
def test_an_aborted_offload_reports_why_it_was_aborted(
    exit_code: ExitCode,
) -> None:
    # A signal that is already aborting keeps every call from starting, which is
    # what an abort arriving mid-batch leaves behind. Driving `_run`
    # directly is what makes the ordering certain: through `offload` the worker
    # may drain the queue before the collecting thread is scheduled to drop it.
    signal = AbortSignal()
    signal.abort(exit_code)
    with pytest.raises(AbortedError) as exc_info:
        _run(ThreadExecutor(workers=1), [partial(_square, 2)], signal)
    assert exc_info.value.exit_code == exit_code


def test_session_offload_runs_a_single_callable() -> None:
    with session() as opened:
        assert opened.offload(partial(add, 3, 4)) == 7


@pytest.mark.timeout(60)
def test_session_offload_runs_a_sequence_at_once_in_the_order_given() -> None:
    # Each job waits for its successor, so the jobs finish in the reverse of the
    # order they were given in, and only run at all if they all run at once.
    count = 5
    finished = [threading.Event() for _ in range(count)]

    def square(index: int) -> int:
        if index + 1 < count:
            assert finished[index + 1].wait(timeout=30)
        finished[index].set()
        return (index + 1) * (index + 1)

    with session() as opened:
        jobs = [partial(square, index) for index in range(count)]
        assert opened.offload(jobs) == (1, 4, 9, 16, 25)


@pytest.mark.timeout(60)
def test_session_offload_runs_no_more_than_limit_callables_at_once() -> None:
    # The jobs meet in pairs, so two must run at once, and a third running
    # beside them would raise the peak.
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=30)
    running = 0
    peak = 0

    def _meet() -> None:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        barrier.wait()
        with lock:
            running -= 1

    with session() as opened:
        opened.offload([_meet] * 4, limit=2)
    assert peak == 2


def test_session_offload_rejects_a_limit_below_one() -> None:
    with session() as opened, pytest.raises(ValueError, match="at least one"):
        opened.offload(partial(_square, 2), limit=0)


def test_session_offload_runs_callables_on_daemon_threads() -> None:
    with session() as opened:
        assert opened.offload(lambda: threading.current_thread().daemon)


def test_session_offload_gives_callables_the_context_of_the_caller() -> None:
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")

    def _offload_from_a_context() -> str:
        marker.set("caller")
        with session() as opened:
            return opened.offload(marker.get)

    assert contextvars.copy_context().run(_offload_from_a_context) == "caller"


def test_session_offload_runs_callables_as_nested_tasks() -> None:
    with session() as opened:
        assert opened.offload(lambda: parent_signal() is not None)


@pytest.mark.timeout(60)
def test_session_offload_raises_the_first_exception_without_waiting() -> None:
    release = threading.Event()

    def _block() -> int:
        assert release.wait(timeout=30)
        return 0

    def _fail() -> int:
        msg = "first"
        raise ValueError(msg)

    with session() as opened:
        try:
            with pytest.raises(ValueError, match="first"):
                opened.offload([_block, _fail])
        finally:
            release.set()


@pytest.mark.timeout(60)
def test_session_offload_aborted_while_a_callable_runs_raises_without_waiting() -> None:
    release = threading.Event()
    with session() as opened:

        def _abort_then_block() -> int:
            opened.abort()
            assert release.wait(timeout=30)
            return 0

        try:
            with pytest.raises(AbortedError) as raised:
                opened.offload(_abort_then_block)
        finally:
            release.set()
    assert raised.value.exit_code is ExitCode.USER_ABORT


def test_session_offload_starts_no_queued_callable_after_an_abort() -> None:
    ran: list[int] = []

    def _record() -> int:
        ran.append(1)
        return 1

    with session() as opened:

        def _abort() -> int:
            opened.abort()
            return 0

        with pytest.raises(AbortedError):
            opened.offload([_abort, _record], limit=1)
    assert not ran


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_session_offload_ended_by_a_base_exception_aborts_the_work_it_started(
    error: type[BaseException],
) -> None:
    seen: list[AbortSignal] = []

    def _record_then_raise() -> int:
        signal = parent_signal()
        assert signal is not None
        seen.append(signal)
        raise error

    with session() as opened:
        with pytest.raises(error):
            opened.offload(_record_then_raise)
        assert seen[0].aborting
        assert seen[0].exit_code is ExitCode.ABORTED


def test_session_offload_on_a_closed_session_raises() -> None:
    with session() as opened:
        pass
    with pytest.raises(WorkflowError):
        opened.offload(partial(_square, 2))
