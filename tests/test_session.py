"""Tests for the ``session`` block and the pools built on it."""

# The pools here are not yet consumed by a run: `pool=` arrives with the entry
# points. What is testable now is the lifetime — that a pool is live while its
# session is, that closing it releases the workers early, and that a session
# releases whatever it created. Liveness is asserted by submitting real work
# through the executor rather than by reading `is_running`, which would also
# pass against an executor that accepts submissions and never runs them.

from __future__ import annotations

import threading
from typing import Any

import pytest

from ropt.components.executors import Submission, ThreadingExecutor, WorkItem
from ropt.exceptions import ExecutorStopped, WorkflowError
from ropt.simple import Session, WorkerPool, serial_pool, session

try:
    import cloudpickle  # ruff: ignore[unused-import]
    import pysqa  # ruff: ignore[unused-import]

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False

# A pool that is not released leaves `collect` waiting on a loop that has gone,
# so a regression here hangs rather than fails.
pytestmark = pytest.mark.timeout(10)


def _run_on(pool: WorkerPool, work: Any) -> Any:
    submission = Submission([WorkItem(function=work)])
    collected: list[Any] = []
    assert pool.executor is not None
    pool.executor.submit(submission)
    submission.collect(lambda item: collected.append(item.result))
    return collected[0]


def _extras(active: Session) -> list[Any]:
    inner = active._session  # ruff: ignore[private-member-access]
    assert inner is not None
    return inner._extras  # ruff: ignore[private-member-access]


def test_thread_pool_runs_work() -> None:
    with session() as active:
        pool = active.thread_pool(workers=2)
        assert _run_on(pool, lambda: 21 * 2) == 42


def test_session_stops_its_pools() -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
    with pytest.raises(ExecutorStopped):
        _run_on(pool, lambda: 1)


def test_pool_close_releases_workers_early() -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
        pool.close()
        with pytest.raises(ExecutorStopped):
            _run_on(pool, lambda: 1)
        # The session outlives the pool it just released.
        assert _run_on(active.thread_pool(workers=1), lambda: 2) == 2


def test_pool_close_is_idempotent() -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
        pool.close()
        pool.close()


def test_pool_context_manager_closes_the_pool() -> None:
    with session() as active:
        with active.thread_pool(workers=1) as pool:
            assert _run_on(pool, lambda: 3) == 3
        with pytest.raises(ExecutorStopped):
            _run_on(pool, lambda: 1)


def test_closing_a_pool_leaves_the_others_running() -> None:
    with session() as active:
        first = active.thread_pool(workers=1)
        second = active.thread_pool(workers=1)
        first.close()
        assert _run_on(second, lambda: 4) == 4


def test_closed_pool_leaves_the_session_extras() -> None:
    # A pool that stays registered after it is closed makes the session's list
    # grow without bound, and is closed a second time at shutdown.
    with session() as active:
        pool = active.thread_pool(workers=1)
        assert _extras(active) == [pool]
        pool.close()
        assert _extras(active) == []


def test_failed_pool_start_leaves_nothing_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An executor that fails to start is not a pool, and must not be released at
    # shutdown. Reachable: a process pool whose worker interpreter never comes up.
    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "no workers for you"
        raise RuntimeError(msg)

    monkeypatch.setattr(ThreadingExecutor, "start", _boom)
    with session() as active:
        with pytest.raises(RuntimeError, match="no workers for you"):
            active.thread_pool(workers=1)
        assert _extras(active) == []


def test_pools_have_independent_batch_ids() -> None:
    with session() as active:
        first = active.thread_pool(workers=1)
        second = active.thread_pool(workers=1)
        assert [first.batch_ids(), first.batch_ids()] == [0, 1]
        assert second.batch_ids() == 0


def test_pool_batch_ids_are_shared_by_its_users() -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
        assert [pool.batch_ids() for _ in range(3)] == [0, 1, 2]


def test_session_refuses_reentry() -> None:
    active = session()
    with pytest.raises(WorkflowError, match="already opened"), active, active:
        pass


def test_closed_session_cannot_be_reopened() -> None:
    active = session()
    with active:
        pass
    with pytest.raises(WorkflowError, match="already opened"), active:
        pass


def test_pool_factory_outside_the_block_raises() -> None:
    active = session()
    with pytest.raises(WorkflowError, match="not open"):
        active.thread_pool(workers=1)


def test_pool_factory_after_the_block_raises() -> None:
    active = session()
    with active:
        pass
    with pytest.raises(WorkflowError, match="not open"):
        active.thread_pool(workers=1)


def test_nested_sessions_are_independent() -> None:
    with session() as outer:
        outer_pool = outer.thread_pool(workers=1)
        with session() as inner:
            assert _run_on(inner.thread_pool(workers=1), lambda: 5) == 5
        # Closing the inner session leaves the outer one untouched.
        assert _run_on(outer_pool, lambda: 6) == 6


def test_pool_built_on_a_driver_thread() -> None:
    # A session is an object, not a context variable, so a thread that holds one
    # can build its own pool. The contextvar form could not do this.
    results: list[Any] = []

    def _build(active: Any) -> None:
        results.append(_run_on(active.thread_pool(workers=1), lambda: 7))

    with session() as active:
        thread = threading.Thread(target=_build, args=(active,))
        thread.start()
        thread.join()
    assert results == [7]


def _double(value: int) -> int:
    return value * 2


@pytest.mark.slow
def test_process_pool_runs_work() -> None:
    from functools import partial  # ruff: ignore[import-outside-top-level]

    with session() as active:
        pool = active.process_pool(workers=1)
        assert _run_on(pool, partial(_double, 21)) == 42


@pytest.mark.skipif(not _TEST_HPC, reason="pysqa or cloudpickle not installed")
def test_hpc_pool_starts_and_stops(tmp_path: Any) -> None:
    with session() as active:
        pool = active.hpc_pool(workers=1, workdir=tmp_path, template="#!/bin/bash\n")
        executor = pool.executor
        assert executor is not None
        assert executor.is_running()
    assert not executor.is_running()


def test_serial_pool_has_no_executor() -> None:
    assert serial_pool().executor is None


def test_serial_pool_needs_no_session() -> None:
    # The free function is the one pool that can be built without a session, so
    # a sessionless run can still be given an explicit batch-ID sequence.
    pool = serial_pool()
    assert [pool.batch_ids(), pool.batch_ids()] == [0, 1]


def test_serial_pool_close_releases_nothing() -> None:
    pool = serial_pool()
    pool.close()
    pool.close()
    assert pool.batch_ids() == 0


def test_serial_pools_have_independent_batch_ids() -> None:
    first, second = serial_pool(), serial_pool()
    assert [first.batch_ids(), first.batch_ids()] == [0, 1]
    assert second.batch_ids() == 0


def test_session_serial_pool_is_released_with_the_session() -> None:
    # A serial pool from a session must be closed with it, so that a run cannot
    # keep drawing batch IDs from a session that has gone.
    with session() as active:
        pool = active.serial_pool()
        assert _extras(active) == [pool]
    assert pool._closed  # ruff: ignore[private-member-access]


def test_session_serial_pool_can_be_closed_early() -> None:
    with session() as active:
        pool = active.serial_pool()
        pool.close()
        assert _extras(active) == []


def test_serial_pool_factory_outside_the_block_raises() -> None:
    with pytest.raises(WorkflowError, match="not open"):
        session().serial_pool()
