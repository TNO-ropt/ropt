"""Tests for the offload method on worker pools."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from functools import partial
from operator import add
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.event_handlers import EventHandler
from ropt.components.executors import ProcessExecutor, ThreadExecutor
from ropt.enums import EnOptEventType
from ropt.exceptions import ExecutionError
from ropt.simple import session

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ropt.components.executors import Executor
    from ropt.events import EnOptEvent
    from ropt.simple import WorkerPool


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


class _OffloadingHandler(EventHandler):
    """Record what offloading from inside a handler returns."""

    def __init__(self, pool: WorkerPool) -> None:
        super().__init__()
        self.pool = pool
        self.outcome: int | None = None

    @property
    def event_types(self) -> set[EnOptEventType]:
        return {EnOptEventType.FINISHED_EVALUATION}

    def _handle_event(self, event: EnOptEvent) -> None:  # ruff: ignore[unused-method-argument]
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
