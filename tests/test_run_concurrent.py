"""Tests for the loop-independent concurrent-job primitive."""

from __future__ import annotations

import threading
from functools import partial

import pytest

from ropt.components.concurrency import run_concurrent


def test_run_concurrent_returns_all_results_in_job_order() -> None:
    count = 5
    # Jobs finish in reversed order.
    finished = [threading.Event() for _ in range(count + 1)]
    finished[count].set()

    def _job(index: int) -> int:
        finished[index + 1].wait(timeout=10.0)
        finished[index].set()
        return index * index

    results = run_concurrent([partial(_job, i) for i in range(count)])
    assert results == [i * i for i in range(count)]


def test_run_concurrent_returns_empty_for_no_jobs() -> None:
    assert run_concurrent([]) == []


def test_run_concurrent_respects_the_concurrency_limit() -> None:
    limit = 3
    barrier = threading.Barrier(limit)
    lock = threading.Lock()
    active = 0
    peak = 0

    def _job() -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=10.0)
        with lock:
            active -= 1
        return 0

    run_concurrent([_job] * (2 * limit), limit=limit)
    assert peak == limit


def test_run_concurrent_propagates_the_first_error_without_awaiting_siblings() -> None:
    release = threading.Event()
    sibling_finished = threading.Event()

    def _fail() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    def _blocking_sibling() -> int:
        release.wait(timeout=10.0)
        sibling_finished.set()
        return 1

    try:
        with pytest.raises(RuntimeError, match="boom"):
            run_concurrent([_fail, _blocking_sibling])
        # The error surfaced while the sibling was still blocked (abandoned).
        assert not sibling_finished.is_set()
    finally:
        release.set()


def test_run_concurrent_skips_pending_jobs_after_a_failure() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def _job(index: int) -> int:
        with lock:
            started.append(index)
            is_first = len(started) == 1
        if is_first:
            msg = "boom"
            raise RuntimeError(msg)
        return index

    with pytest.raises(RuntimeError, match="boom"):
        run_concurrent([partial(_job, i) for i in range(5)], limit=1)
    # With limit=1 the first job to run fails and sets the stop flag before
    # releasing its slot, so no pending job is ever started.
    assert len(started) == 1
