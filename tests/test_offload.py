"""Tests for the offload executor-dispatch helper."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import threading
from functools import partial
from operator import add
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.event_handlers import EventDispatcher, EventHandler
from ropt.components.executors import ThreadingExecutor
from ropt.enums import EnOptEventType
from ropt.exceptions import ExecutorFailure, ExecutorStopped, WorkflowError
from ropt.simple import (
    can_offload,
    handlers,
    offload,
    optimize,
    processes,
    threads,
)
from ropt.simple._session import _Session, current_session
from ropt.simple.compose import current_executor

if TYPE_CHECKING:
    from collections.abc import Callable

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


def test_offload_raises_without_a_block() -> None:
    with pytest.raises(WorkflowError, match="execution block"):
        offload(partial(add, 3, 4))


def test_offload_sequence_raises_without_a_block() -> None:
    with pytest.raises(WorkflowError, match="execution block"):
        offload([partial(_square, 1), partial(_square, 2)])


def test_offload_empty_sequence_raises_without_a_block() -> None:
    with pytest.raises(WorkflowError, match="execution block"):
        offload([])


def test_offload_empty_sequence_returns_empty() -> None:
    with threads(workers=1):
        assert offload([]) == ()


def test_can_offload_is_false_without_a_block() -> None:
    assert can_offload() is False


def test_can_offload_is_true_within_a_block() -> None:
    with threads(workers=1):
        assert can_offload() is True


def test_can_offload_false_for_stopped_executor(monkeypatch: Any) -> None:
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
def test_dying_worker_reported_to_offload_caller() -> None:
    with pytest.raises(ExecutorFailure, match="killed"), processes(workers=1):
        offload(_kill_worker)


def test_offload_preserves_order_across_workers() -> None:
    # Each job waits for its successor, so the jobs finish in exactly the
    # reverse of the order they were submitted in and a result tuple in
    # submission order can only come from reordering by index. Staggered sleeps
    # only make that reversal likely: a scramble under load would let results
    # returned in completion order pass. One worker per job, or the chain
    # deadlocks on the pool.
    count = 5
    finished = [threading.Event() for _ in range(count)]

    def square(index: int) -> int:
        if index + 1 < count:
            assert finished[index + 1].wait(timeout=30)
        finished[index].set()
        return (index + 1) * (index + 1)

    with threads(workers=count):
        jobs = [partial(square, index) for index in range(count)]
        assert offload(jobs) == (1, 4, 9, 16, 25)


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


def test_can_offload_false_on_session_loop() -> None:
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


def test_offload_from_unrelated_event_loop() -> None:
    async def _offload_in_a_cell() -> tuple[bool, int]:  # ruff: ignore[unused-async]
        return can_offload(), offload(partial(_square, 4))

    with threads(workers=1):
        assert asyncio.run(_offload_in_a_cell()) == (True, 16)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("work", [_exit_process, _interrupt])
def test_offload_base_exception_reaches_caller(
    work: Callable[[], int],
) -> None:
    with pytest.raises(BaseException) as raised, threads(workers=2):  # ruff: ignore[pytest-raises-too-broad]
        offload(work)
    assert not isinstance(raised.value, Exception)


@pytest.mark.timeout(30)
def test_dying_session_stops_block_executor() -> None:
    captured: list[Any] = []
    with pytest.raises(SystemExit), threads(workers=2):  # ruff: ignore[pytest-raises-with-multiple-statements]
        captured.append(current_executor())
        offload(_exit_process)
    assert not captured[0].is_running()


@pytest.mark.timeout(30)
def test_dying_session_leaves_no_unretrieved_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # asyncio reports this through logging, not warnings:
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        with pytest.raises(SystemExit), threads(workers=2):
            offload(_exit_process)
        gc.collect()
    assert "never retrieved" not in caplog.text


@pytest.mark.timeout(30)
def test_block_on_dead_session_reports_stopped() -> None:
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
    assert "is not running" in str(captured[0])


def test_shutdown_race_reports_stopped_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()
    session.start()
    session.stop()
    # Only a race reaches the late translation, so every check before the work
    # is handed to the loop has to see a session that is still alive: one in
    # `_require_task_group`, one at the top of `_start_on_loop`. The loop is
    # already closed, so the hand-off raises RuntimeError, and the next check --
    # the one inside the handler -- reports the truth. Feeding one `False` too
    # few stops at the up-front check and never reaches the handler at all.
    seen = iter([False, False])
    monkeypatch.setattr(
        session._stopped,  # ruff: ignore[private-member-access]
        "is_set",
        lambda: next(seen, True),
    )
    with pytest.raises(WorkflowError, match="is not running"):
        session.open_dispatcher(EventDispatcher())


class _OffloadingHandler(EventHandler):
    """Record what `offload` does when called from inside a handler."""

    def __init__(self) -> None:
        super().__init__()
        self.outcome: str | None = None
        self.allowed: bool | None = None

    @property
    def event_types(self) -> set[EnOptEventType]:
        return {EnOptEventType.FINISHED_EVALUATION}

    def handle_event(self, event: EnOptEvent) -> None:  # ruff: ignore[unused-method-argument]
        if self.outcome is not None:
            return
        self.allowed = can_offload()
        try:
            self.outcome = f"returned {offload(partial(_square, 4))}"
        except WorkflowError as exc:
            self.outcome = f"raised {exc}"


def _run_one(**kwargs: Any) -> None:
    config = {
        "variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6},
        "optimizer": {"max_functions": 2},
    }
    optimize(
        config, np.zeros(2), lambda variables, _: float(np.sum(variables**2)), **kwargs
    )


@pytest.mark.timeout(60)
def test_inline_handler_in_shared_block_cannot_offload() -> None:
    handler = _OffloadingHandler()
    with threads(workers=2), handlers(handler):
        _run_one()
    assert handler.allowed is False
    assert handler.outcome is not None
    assert "event loop" in handler.outcome


@pytest.mark.timeout(60)
def test_threaded_handler_in_shared_block_cannot_offload() -> None:
    # A threaded handler runs on a dispatcher worker, which carries no session,
    # so it is refused for a different reason than the inline one -- but it is
    # refused, and `can_offload` says so before it is tried.
    handler = _OffloadingHandler()
    with threads(workers=2), handlers(threaded=handler):
        _run_one()
    assert handler.allowed is False
    assert handler.outcome is not None
    assert "no executor" in handler.outcome
    # A block *is* open here, so "open an execution block" on its own would
    # send the reader to fix something that is not broken.
    assert "handler running in a thread" in handler.outcome


@pytest.mark.timeout(60)
def test_local_handler_can_offload() -> None:
    # A handler passed to a single `optimize` call is not on the event loop: it
    # runs on the thread driving the run, where the block is open as usual.
    handler = _OffloadingHandler()
    with threads(workers=2):
        _run_one(handlers=[handler])
    assert handler.allowed is True
    assert handler.outcome == "returned 16"
