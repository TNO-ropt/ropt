"""Tests for the offload pool-dispatch helper."""

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
from ropt.enums import EnOptEventType
from ropt.exceptions import ExecutorFailure, WorkflowError
from ropt.simple import WorkerPool, offload, optimize, serial_pool, session
from ropt.simple._session import _Session

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.events import EnOptEvent
    from ropt.simple import Session


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


def test_offload_without_a_pool_runs_inline() -> None:
    # No pool used to mean "no execution block is open", which offload()
    # refused. Now it just means "run here", so the call succeeds inline.
    assert offload(partial(add, 3, 4)) == 7


def test_offload_sequence_without_a_pool_runs_inline() -> None:
    assert offload([partial(_square, 1), partial(_square, 2)]) == (1, 4)


def test_offload_empty_sequence_without_a_pool_returns_empty() -> None:
    assert offload([]) == ()


def test_offload_with_a_serial_pool_runs_inline() -> None:
    # A serial pool has no executor, so it is the explicit spelling of "run
    # inline" -- distinct from passing no pool at all, but behaviourally the
    # same.
    assert offload(partial(add, 3, 4), pool=serial_pool()) == 7


def test_offload_empty_sequence_returns_empty_with_a_pool() -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
        assert offload([], pool=pool) == ()


def test_offload_single_call_with_a_thread_pool() -> None:
    with session() as active:
        pool = active.thread_pool(workers=2)
        assert offload(partial(add, 3, 4), pool=pool) == 7


def test_offload_sequence_with_a_thread_pool() -> None:
    with session() as active:
        pool = active.thread_pool(workers=3)
        assert offload([partial(_square, i) for i in range(1, 6)], pool=pool) == (
            1,
            4,
            9,
            16,
            25,
        )


def test_offload_runs_different_functions() -> None:
    with session() as active:
        pool = active.thread_pool(workers=2)
        assert offload([partial(_square, 3), partial(_double, 5)], pool=pool) == (
            9,
            10,
        )


def test_offload_sequence_with_a_process_pool() -> None:
    with session() as active:
        pool = active.process_pool(workers=2)
        assert offload([partial(_square, i) for i in (1, 2, 3)], pool=pool) == (
            1,
            4,
            9,
        )


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_dying_worker_reported_to_offload_caller() -> None:
    with pytest.raises(ExecutorFailure, match="killed"), session() as active:
        offload(_kill_worker, pool=active.process_pool(workers=1))


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

    with session() as active:
        pool = active.thread_pool(workers=count)
        jobs = [partial(square, index) for index in range(count)]
        assert offload(jobs, pool=pool) == (1, 4, 9, 16, 25)


def test_offload_raises_on_the_pools_event_loop() -> None:
    async def _offload_on_loop(pool: WorkerPool) -> int:  # ruff: ignore[unused-async]
        return offload(partial(_square, 5), pool=pool)

    with session() as active:
        pool = active.thread_pool(workers=1)
        inner = active._session  # ruff: ignore[private-member-access]
        assert inner is not None
        assert inner._loop is not None  # ruff: ignore[private-member-access]
        future = asyncio.run_coroutine_threadsafe(
            _offload_on_loop(pool),
            inner._loop,  # ruff: ignore[private-member-access]
        )
        with pytest.raises(WorkflowError, match="event loop"):
            future.result(timeout=5)


def test_offload_from_unrelated_event_loop() -> None:
    async def _offload_in_a_cell(pool: WorkerPool) -> int:  # ruff: ignore[unused-async]
        return offload(partial(_square, 4), pool=pool)

    with session() as active:
        pool = active.thread_pool(workers=1)
        assert asyncio.run(_offload_in_a_cell(pool)) == 16


@pytest.mark.timeout(30)
@pytest.mark.parametrize("work", [_exit_process, _interrupt])
def test_offload_base_exception_reaches_caller(
    work: Callable[[], int],
) -> None:
    with pytest.raises(BaseException) as raised, session() as active:  # ruff: ignore[pytest-raises-too-broad]
        offload(work, pool=active.thread_pool(workers=2))
    assert not isinstance(raised.value, Exception)


@pytest.mark.timeout(30)
def test_dying_session_stops_the_pool() -> None:
    with pytest.raises(SystemExit), session() as active:  # ruff: ignore[pytest-raises-with-multiple-statements]
        pool = active.thread_pool(workers=2)
        offload(_exit_process, pool=pool)
    assert pool.executor is not None
    assert not pool.executor.is_running()


@pytest.mark.timeout(30)
def test_dying_session_leaves_no_unretrieved_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # asyncio reports this through logging, not warnings:
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        with pytest.raises(SystemExit), session() as active:
            offload(_exit_process, pool=active.thread_pool(workers=2))
        gc.collect()
    assert "never retrieved" not in caplog.text


@pytest.mark.timeout(30)
def test_group_on_dead_session_reports_stopped() -> None:
    captured: list[BaseException] = []

    def _reopen_after_the_session_dies(active: Session) -> None:
        pool = active.thread_pool(workers=2)
        with pytest.raises(SystemExit):
            offload(_exit_process, pool=pool)
        try:
            active.shared_handlers()
        except BaseException as exc:  # ruff: ignore[blind-except]
            captured.append(exc)

    # The block exit re-raises the failure that killed the session, so what the
    # reopen attempt raised has to be carried out of the block to be asserted.
    with pytest.raises(SystemExit), session() as active:
        _reopen_after_the_session_dies(active)
    assert len(captured) == 1
    assert isinstance(captured[0], WorkflowError)
    assert "is not running" in str(captured[0])


def test_shutdown_race_reports_stopped_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sess = _Session()
    sess.start()
    sess.stop()
    # Only a race reaches the late translation, so every check before the work
    # is handed to the loop has to see a session that is still alive: one in
    # `_require_task_group`, one at the top of `_start_on_loop`. The loop is
    # already closed, so the hand-off raises RuntimeError, and the next check --
    # the one inside the handler -- reports the truth. Feeding one `False` too
    # few stops at the up-front check and never reaches the handler at all.
    seen = iter([False, False])
    monkeypatch.setattr(
        sess._stopped,  # ruff: ignore[private-member-access]
        "is_set",
        lambda: next(seen, True),
    )
    with pytest.raises(WorkflowError, match="is not running"):
        sess.open_dispatcher(EventDispatcher())


class _OffloadingHandler(EventHandler):
    """Record what `offload` does when called from inside a handler."""

    def __init__(self) -> None:
        super().__init__()
        self.pool: WorkerPool | None = None
        self.outcome: str | None = None

    @property
    def event_types(self) -> set[EnOptEventType]:
        return {EnOptEventType.FINISHED_EVALUATION}

    def _handle_event(self, event: EnOptEvent) -> None:  # ruff: ignore[unused-method-argument]
        if self.outcome is not None:
            return
        try:
            self.outcome = f"returned {offload(partial(_square, 4), pool=self.pool)}"
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
def test_inline_handler_in_shared_group_cannot_offload() -> None:
    handler = _OffloadingHandler()
    with session() as active:
        group = active.shared_handlers(handler)
        handler.pool = active.thread_pool(workers=2)
        _run_one(pool=handler.pool, handlers=[group])
    assert handler.outcome is not None
    assert "event loop" in handler.outcome


@pytest.mark.timeout(60)
def test_threaded_handler_in_shared_group_can_offload() -> None:
    # A threaded handler runs on a dispatcher worker thread, not the session's
    # own loop thread, so a pool sharing that session is not "its own loop"
    # from there: the offload dispatches and returns a real result.
    handler = _OffloadingHandler()
    with session() as active:
        group = active.shared_handlers(threaded=handler)
        handler.pool = active.thread_pool(workers=2)
        _run_one(pool=handler.pool, handlers=[group])
    assert handler.outcome == "returned 16"


@pytest.mark.timeout(60)
def test_local_handler_can_offload() -> None:
    # A handler passed straight to `optimize` is not on any event loop: it runs
    # on the thread driving the run, where the pool it is given works as usual.
    handler = _OffloadingHandler()
    with session() as active:
        pool = active.thread_pool(workers=2)
        handler.pool = pool
        _run_one(pool=pool, handlers=[handler])
    assert handler.outcome == "returned 16"
