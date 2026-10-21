"""Tests for running arbitrary callables with `run`."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from functools import partial
from operator import add
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pytest

from ropt import optimize, run, session
from ropt._concurrency import AbortScope, current_abort_scope
from ropt._executors import ProcessExecutor, ThreadExecutor
from ropt._handlers import EventHandler
from ropt._running._work import _run
from ropt.enums import EnOptEventType, ExitCode
from ropt.exceptions import AbortedError, ExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ropt import WorkerPool
    from ropt._executors import Executor
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


def test_pool_run_empty_sequence_returns_empty(
    pools: Callable[..., WorkerPool],
) -> None:
    assert run([], pool=pools(workers=1)) == ()


def test_pool_run_single_call_with_a_thread_pool(
    pools: Callable[..., WorkerPool],
) -> None:
    assert run(partial(add, 3, 4), pool=pools(workers=2)) == 7


class _ShutdownPool:
    # `concurrent.futures` refuses new work with `RuntimeError` once the
    # interpreter is shutting down.
    def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

    @staticmethod
    def submit(*_args: Any, **_kwargs: Any) -> None:
        msg = "cannot schedule new futures after interpreter shutdown"
        raise RuntimeError(msg)

    def shutdown(self, *_args: Any, **_kwargs: Any) -> None: ...


def test_pool_run_on_a_shut_down_pool_raises_aborted_with_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, pools: Callable[..., WorkerPool]
) -> None:
    monkeypatch.setattr(
        "ropt._executors._thread_executor.ThreadPoolExecutor",
        _ShutdownPool,
    )
    with pytest.raises(AbortedError) as raised:
        run(partial(_square, 3), pool=pools(workers=1))
    assert raised.value.exit_code is ExitCode.EXECUTOR_SHUT_DOWN


def test_pool_run_sequence_with_a_thread_pool(
    pools: Callable[..., WorkerPool],
) -> None:
    assert run([partial(_square, i) for i in range(1, 6)], pool=pools(workers=3)) == (
        1,
        4,
        9,
        16,
        25,
    )


def test_pool_run_sequence_of_different_functions(
    pools: Callable[..., WorkerPool],
) -> None:
    assert run([partial(_square, 3), partial(_double, 5)], pool=pools(workers=2)) == (
        9,
        10,
    )


@pytest.mark.slow
def test_pool_run_sequence_with_a_process_pool(
    pools: Callable[..., WorkerPool],
) -> None:
    assert run(
        [partial(_square, i) for i in (1, 2, 3)], pool=pools(ProcessExecutor, workers=2)
    ) == (
        1,
        4,
        9,
    )


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_dying_worker_reported_to_pool_run_caller(
    pools: Callable[..., WorkerPool],
) -> None:
    with pytest.raises(ExecutionError, match="could not be run"):
        run(_kill_worker, pool=pools(ProcessExecutor, workers=1))


def test_pool_run_preserves_order_across_workers(
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
    assert run(jobs, pool=pools(workers=count)) == (1, 4, 9, 16, 25)


def test_pool_run_from_an_event_loop(pools: Callable[..., WorkerPool]) -> None:
    # Nothing in the pools touches asyncio, so a call from inside a
    # coroutine -- a notebook cell, say -- is an ordinary blocking call.
    async def _run_in_a_cell(pool: WorkerPool) -> int:  # ruff: ignore[unused-async]
        return run(partial(_square, 4), pool=pool)

    assert asyncio.run(_run_in_a_cell(pools(workers=1))) == 16


@pytest.mark.timeout(30)
@pytest.mark.parametrize("work", [_exit_process, _interrupt])
def test_pool_run_base_exception_reaches_caller(
    pools: Callable[..., WorkerPool], work: Callable[[], int]
) -> None:
    # A worker thread cannot exit the interpreter on its own, so the exception
    # is delivered to the caller, which is where it means something.
    with pytest.raises((SystemExit, KeyboardInterrupt)):
        run(work, pool=pools(workers=2))


@pytest.mark.timeout(30)
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_a_pool_run_ended_by_a_base_exception_aborts_the_work_it_started(
    pools: Callable[..., WorkerPool], error: type[BaseException]
) -> None:
    # Work started from the function takes this scope as its parent,
    # so whether it aborts is what reaches that work.
    seen: list[AbortScope] = []

    def _record_then_raise() -> int:
        scope = current_abort_scope()
        assert scope is not None
        seen.append(scope)
        raise error

    with pytest.raises(error):
        run(_record_then_raise, pool=pools(workers=1))
    assert seen[0].aborting
    assert seen[0].exit_code is ExitCode.ABORTED


class _RunningHandler(EventHandler):
    """Record what calling `run` from inside a handler returns."""

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
            self.outcome = run(partial(_square, 4), pool=self.pool)


@pytest.mark.timeout(60)
def test_handler_can_call_pool_run(pools: Callable[..., WorkerPool]) -> None:
    # A handler runs on the thread driving the optimization, not on a worker,
    # so the pool it is given works there as usual.
    pool = pools(workers=2)
    handler = _RunningHandler(pool)
    config = {
        "variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6},
        "optimizer": {"max_functions": 2},
    }
    optimize(
        config,
        np.zeros(2),
        lambda variables, _: float(np.sum(variables**2)),
        pool=pool,
        handlers=[handler],
    )
    assert handler.outcome == 16


@pytest.mark.timeout(60)
def test_a_pool_run_aborted_while_its_function_runs_raises() -> None:
    # The function is already on a worker when the abort arrives. A worker
    # cannot be interrupted, and `run` does not wait for it.
    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def _abort_then_return() -> int:
            opened.abort()
            return 7

        with pytest.raises(AbortedError) as raised:
            run(_abort_then_return, pool=pool)
    assert raised.value.exit_code is ExitCode.USER_ABORT


@pytest.mark.parametrize("exit_code", [ExitCode.ABORTED, ExitCode.ABORTED_ON_ERROR])
def test_an_aborted_pool_run_reports_why_it_was_aborted(
    exit_code: ExitCode,
) -> None:
    # A scope that is already aborting keeps every call from starting, which is
    # what an abort arriving mid-batch leaves behind. Driving `_run`
    # directly is what makes the ordering certain: through `run` the worker
    # may drain the queue before the collecting thread is scheduled to drop it.
    scope = AbortScope()
    scope.abort(exit_code)
    with pytest.raises(AbortedError) as exc_info:
        _run(ThreadExecutor(workers=1), [partial(_square, 2)], scope)
    assert exc_info.value.exit_code == exit_code


def test_run_without_a_pool_runs_a_single_callable() -> None:
    assert run(partial(add, 3, 4)) == 7


def test_run_without_a_pool_runs_a_sequence_in_turn_in_the_order_given() -> None:
    order: list[int] = []

    def square(index: int) -> int:
        order.append(index)
        return index * index

    assert run([partial(square, index) for index in range(1, 6)]) == (
        1,
        4,
        9,
        16,
        25,
    )
    assert order == [1, 2, 3, 4, 5]


def test_run_without_a_pool_returns_an_empty_tuple_for_an_empty_sequence() -> None:
    assert run([]) == ()


def test_run_without_a_pool_runs_callables_on_the_calling_thread() -> None:
    caller = threading.current_thread()
    assert run([threading.current_thread] * 2) == (caller, caller)


def test_run_without_a_pool_runs_callables_in_a_nested_scope() -> None:
    with session():
        outside = current_abort_scope()
        inside = run(current_abort_scope)
    assert inside is not None
    assert inside is not outside


def test_run_without_a_pool_starts_no_callable_after_a_failure() -> None:
    ran: list[int] = []
    seen: list[AbortScope] = []

    def _fail() -> int:
        scope = current_abort_scope()
        assert scope is not None
        seen.append(scope)
        msg = "first"
        raise ValueError(msg)

    def _record() -> int:
        ran.append(1)
        return 1

    with pytest.raises(ValueError, match="first"):
        run([_fail, _record])
    assert not ran
    assert seen[0].exit_code is ExitCode.ABORTED_ON_ERROR


def test_run_without_a_pool_starts_no_callable_after_an_abort() -> None:
    ran: list[int] = []

    def _record() -> int:
        ran.append(1)
        return 1

    with session() as opened:

        def _abort() -> int:
            opened.abort()
            return 0

        with pytest.raises(AbortedError) as raised:
            run([_abort, _record])
    assert raised.value.exit_code is ExitCode.USER_ABORT
    assert not ran


def test_run_without_a_pool_raises_when_the_last_callable_was_aborted() -> None:
    # The callable returns normally, but its value may stand for an early exit.
    with session() as opened:

        def _abort() -> int:
            opened.abort()
            return 0

        with pytest.raises(AbortedError) as raised:
            run(_abort)
    assert raised.value.exit_code is ExitCode.USER_ABORT


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_run_without_a_pool_ended_by_a_base_exception_aborts_the_work_it_started(
    error: type[BaseException],
) -> None:
    seen: list[AbortScope] = []

    def _record_then_raise() -> int:
        scope = current_abort_scope()
        assert scope is not None
        seen.append(scope)
        raise error

    with pytest.raises(error):
        run(_record_then_raise)
    assert seen[0].aborting
    assert seen[0].exit_code is ExitCode.ABORTED
