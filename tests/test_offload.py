"""Tests for the offload executor-dispatch helper."""

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
from ropt.enums import EnOptEventType
from ropt.exceptions import ExecutionError, WorkflowError
from ropt.simple import ProcessExecutor, ThreadExecutor, offload, optimize

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.components.executors import Executor
    from ropt.events import EnOptEvent


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


def test_offload_without_an_executor_runs_inline() -> None:
    # Passing no executor means "run here", so a call site that may or may not
    # have one needs no fallback of its own.
    assert offload(partial(add, 3, 4)) == 7


def test_offload_sequence_without_an_executor_runs_inline() -> None:
    assert offload([partial(_square, 1), partial(_square, 2)]) == (1, 4)


def test_offload_empty_sequence_without_an_executor_returns_empty() -> None:
    assert offload([]) == ()


def test_offload_empty_sequence_returns_empty_with_an_executor() -> None:
    with ThreadExecutor(workers=1) as executor:
        assert offload([], executor=executor) == ()


def test_offload_single_call_with_a_thread_executor() -> None:
    with ThreadExecutor(workers=2) as executor:
        assert offload(partial(add, 3, 4), executor=executor) == 7


def test_offload_sequence_with_a_thread_executor() -> None:
    with ThreadExecutor(workers=3) as executor:
        assert offload(
            [partial(_square, i) for i in range(1, 6)], executor=executor
        ) == (1, 4, 9, 16, 25)


def test_offload_sequence_of_different_functions() -> None:
    with ThreadExecutor(workers=2) as executor:
        assert offload(
            [partial(_square, 3), partial(_double, 5)], executor=executor
        ) == (9, 10)


@pytest.mark.slow
def test_offload_sequence_with_a_process_executor() -> None:
    with ProcessExecutor(workers=2) as executor:
        assert offload([partial(_square, i) for i in (1, 2, 3)], executor=executor) == (
            1,
            4,
            9,
        )


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_dying_worker_reported_to_offload_caller() -> None:
    with (
        ProcessExecutor(workers=1) as executor,
        pytest.raises(ExecutionError, match="could not be run"),
    ):
        offload(_kill_worker, executor=executor)


def test_offload_preserves_order_across_workers() -> None:
    # Each job waits for its successor, so the jobs finish in exactly the
    # reverse of the order they were submitted in and a result tuple in
    # submission order can only come from reordering by index. One worker per
    # job, or the chain deadlocks on the executor.
    count = 5
    finished = [threading.Event() for _ in range(count)]

    def square(index: int) -> int:
        if index + 1 < count:
            assert finished[index + 1].wait(timeout=30)
        finished[index].set()
        return (index + 1) * (index + 1)

    with ThreadExecutor(workers=count) as executor:
        jobs = [partial(square, index) for index in range(count)]
        assert offload(jobs, executor=executor) == (1, 4, 9, 16, 25)


def test_offload_from_an_event_loop() -> None:
    # Nothing in the executors touches asyncio, so a call from inside a
    # coroutine -- a notebook cell, say -- is an ordinary blocking call.
    async def _offload_in_a_cell(executor: Executor) -> int:  # ruff: ignore[unused-async]
        return offload(partial(_square, 4), executor=executor)

    with ThreadExecutor(workers=1) as executor:
        assert asyncio.run(_offload_in_a_cell(executor)) == 16


@pytest.mark.timeout(30)
@pytest.mark.parametrize("work", [_exit_process, _interrupt])
def test_offload_base_exception_reaches_caller(work: Callable[[], int]) -> None:
    # A worker thread cannot exit the interpreter on its own, so the exception
    # is delivered to the caller, which is where it means something.
    with (
        ThreadExecutor(workers=2) as executor,
        pytest.raises((SystemExit, KeyboardInterrupt)),
    ):
        offload(work, executor=executor)


class _OffloadingHandler(EventHandler):
    """Record what `offload` does when called from inside a handler."""

    def __init__(self) -> None:
        super().__init__()
        self.executor: Executor | None = None
        self.outcome: str | None = None

    @property
    def event_types(self) -> set[EnOptEventType]:
        return {EnOptEventType.FINISHED_EVALUATION}

    def _handle_event(self, event: EnOptEvent) -> None:  # ruff: ignore[unused-method-argument]
        if self.outcome is not None:
            return
        try:
            offloaded = offload(partial(_square, 4), executor=self.executor)
        except WorkflowError as exc:
            self.outcome = f"raised {exc}"
        else:
            self.outcome = f"returned {offloaded}"


def _run_one(**kwargs: Any) -> None:
    config = {
        "variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6},
        "optimizer": {"max_functions": 2},
    }
    optimize(
        config, np.zeros(2), lambda variables, _: float(np.sum(variables**2)), **kwargs
    )


@pytest.mark.timeout(60)
def test_handler_can_offload() -> None:
    # A handler runs on the thread driving the run, not on a worker, so the
    # executor it is given works there as usual.
    handler = _OffloadingHandler()
    with ThreadExecutor(workers=2) as executor:
        handler.executor = executor
        _run_one(executor=executor, handlers=[handler])
    assert handler.outcome == "returned 16"
