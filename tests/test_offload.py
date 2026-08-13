"""Tests for the offload executor-dispatch helper."""

from __future__ import annotations

import asyncio
import time
from functools import partial
from operator import add

import pytest

from ropt.exceptions import WorkflowError
from ropt.simple import can_offload, offload, processes, threads
from ropt.simple.compose import current_session


def _square(x: int) -> int:
    return x * x


def _double(x: int) -> int:
    return x + x


def _slow_square(x: int) -> int:
    # Later inputs finish first, so a correct offload must reorder by index.
    time.sleep(0.02 * (6 - x))
    return x * x


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
