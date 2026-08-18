"""Tests for the offload executor-dispatch helper."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import time
from functools import partial
from operator import add
from typing import TYPE_CHECKING, Any

import pytest

from ropt.components.event_handlers import EventDispatcher
from ropt.components.executors import ThreadingExecutor
from ropt.exceptions import ExecutorFailure, ExecutorStopped, WorkflowError
from ropt.simple import can_offload, handlers, offload, processes, threads
from ropt.simple._session import Session
from ropt.simple.compose import current_executor, current_session

if TYPE_CHECKING:
    from collections.abc import Callable


def _square(x: int) -> int:
    return x * x


def _double(x: int) -> int:
    return x + x


def _slow_square(x: int) -> int:
    # Later inputs finish first, so a correct offload must reorder by index.
    time.sleep(0.02 * (6 - x))
    return x * x


def _exit_process() -> int:
    sys.exit(3)


def _interrupt() -> int:
    raise KeyboardInterrupt


def _kill_worker() -> int:
    os._exit(1)


def test_offload_raises_without_a_block() -> None:
    with pytest.raises(WorkflowError, match="execution block"):
        offload(partial(add, 3, 4))


def test_offload_sequence_raises_without_a_block() -> None:
    with pytest.raises(WorkflowError, match="execution block"):
        offload([partial(_square, 1), partial(_square, 2)])


def test_offload_empty_sequence_returns_empty() -> None:
    assert offload([]) == ()


def test_can_offload_is_false_without_a_block() -> None:
    assert can_offload() is False


def test_can_offload_is_true_within_a_block() -> None:
    with threads(workers=1):
        assert can_offload() is True


def test_that_can_offload_is_false_for_a_stopped_executor(monkeypatch: Any) -> None:
    # The documented fallback only works if can_offload() reports False exactly
    # when offload() would refuse the work.
    async def _start_and_stop() -> ThreadingExecutor:
        executor = ThreadingExecutor(workers=1)
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.cancel()
        return executor

    stopped = asyncio.run(_start_and_stop())
    monkeypatch.setattr("ropt.simple._offload.current_executor", lambda: stopped)
    assert can_offload() is False
    with pytest.raises(ExecutorStopped):
        offload(partial(add, 3, 4))


def test_offload_single_call_under_threads() -> None:
    with threads(workers=2):
        assert offload(partial(add, 3, 4)) == 7


def test_offload_sequence_under_threads() -> None:
    with threads(workers=3):
        assert offload([partial(_square, i) for i in range(1, 6)]) == (1, 4, 9, 16, 25)


def test_offload_runs_different_functions() -> None:
    with threads(workers=2):
        assert offload([partial(_square, 3), partial(_double, 5)]) == (9, 10)


def test_offload_sequence_under_processes() -> None:
    with processes(workers=2):
        assert offload([partial(_square, i) for i in (1, 2, 3)]) == (1, 4, 9)


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_that_a_worker_process_dying_is_reported_to_the_offload_caller() -> None:
    with pytest.raises(ExecutorFailure, match="killed"), processes(workers=1):
        offload(_kill_worker)


def test_offload_preserves_order_across_workers() -> None:
    with threads(workers=5):
        functions = [partial(_slow_square, i) for i in range(1, 6)]
        assert offload(functions) == (1, 4, 9, 16, 25)


def test_offload_raises_on_the_event_loop() -> None:
    async def _offload_on_loop() -> int:  # ruff: ignore[unused-async]
        return offload(partial(_square, 5))

    with threads(workers=1):
        session = current_session()
        assert session is not None
        assert session._loop is not None  # ruff: ignore[private-member-access]
        future = asyncio.run_coroutine_threadsafe(
            _offload_on_loop(),
            session._loop,  # ruff: ignore[private-member-access]
        )
        with pytest.raises(WorkflowError, match="event loop"):
            future.result(timeout=5)


def test_that_can_offload_is_false_on_the_session_event_loop() -> None:
    async def _check_on_loop() -> bool:  # ruff: ignore[unused-async]
        return can_offload()

    with threads(workers=1):
        session = current_session()
        assert session is not None
        assert session._loop is not None  # ruff: ignore[private-member-access]
        future = asyncio.run_coroutine_threadsafe(
            _check_on_loop(),
            session._loop,  # ruff: ignore[private-member-access]
        )
        assert future.result(timeout=5) is False


def test_that_offloading_from_an_unrelated_event_loop_still_dispatches() -> None:
    async def _offload_in_a_cell() -> tuple[bool, int]:  # ruff: ignore[unused-async]
        return can_offload(), offload(partial(_square, 4))

    with threads(workers=1):
        assert asyncio.run(_offload_in_a_cell()) == (True, 16)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("work", [_exit_process, _interrupt])
def test_that_a_base_exception_in_offloaded_work_reaches_the_caller(
    work: Callable[[], int],
) -> None:
    with pytest.raises(BaseException) as raised, threads(workers=2):  # ruff: ignore[pytest-raises-too-broad]
        offload(work)
    assert not isinstance(raised.value, Exception)


@pytest.mark.timeout(30)
def test_that_a_block_torn_down_by_a_dying_session_still_stops_its_executor() -> None:
    captured: list[Any] = []
    with pytest.raises(SystemExit), threads(workers=2):  # ruff: ignore[pytest-raises-with-multiple-statements]
        captured.append(current_executor())
        offload(_exit_process)
    assert not captured[0].is_running()


@pytest.mark.timeout(30)
def test_that_a_dying_session_leaves_no_unretrieved_task_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # asyncio reports this through logging, not warnings:
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        with pytest.raises(SystemExit), threads(workers=2):
            offload(_exit_process)
        gc.collect()
    assert "never retrieved" not in caplog.text


@pytest.mark.timeout(30)
def test_that_opening_a_block_on_a_dead_session_says_the_session_stopped() -> None:
    captured: list[BaseException] = []

    def _reopen_after_the_session_dies() -> None:
        with pytest.raises(SystemExit):
            offload(_exit_process)
        try:
            with handlers():
                pass
        except BaseException as exc:  # ruff: ignore[blind-except]
            captured.append(exc)

    # The block exit re-raises the failure that killed the session, so what the
    # reopen attempt raised has to be carried out of the block to be asserted.
    with pytest.raises(SystemExit), threads(workers=2):
        _reopen_after_the_session_dies()
    assert len(captured) == 1
    assert isinstance(captured[0], WorkflowError)
    assert "cannot be reused" in str(captured[0])


def test_that_losing_the_race_with_the_shutdown_reports_a_stopped_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session()
    session.start()
    session.stop()
    # Only a race reaches the late translation, so let the up-front check see a
    # session that is still alive and the loop report the truth.
    seen = iter([False])
    monkeypatch.setattr(
        session._stopped,  # ruff: ignore[private-member-access]
        "is_set",
        lambda: next(seen, True),
    )
    with pytest.raises(WorkflowError, match="cannot be reused"):
        session.open_dispatcher(EventDispatcher())
