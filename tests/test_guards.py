"""Tests for how the entry points behave with a pool that cannot serve them."""

# Every case is asserted at every entry point that takes `pool=`. The entry
# points do not share a single code path, so a behaviour that changes for one is
# easy to miss in another, and the parametrization is what makes that visible.
# test_live_pool_accepted is the control: without it a refusal test would
# still pass if the entry point had stopped working for any pool at all.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.exceptions import ExecutionError, WorkflowError
from ropt.simple import (
    HistoryHandler,
    evaluate,
    evaluate_batch,
    offload,
    optimize,
    optimize_many,
    session,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

    from ropt.simple import EvaluationFunctionContext, Session, WorkerPool


@pytest.fixture(name="opened")
def opened_fixture() -> Iterator[Session]:
    with session() as opened:
        yield opened


_CONFIG: dict[str, Any] = {
    "optimizer": {"max_functions": 2},
    "variables": {"variable_count": 2, "perturbation_magnitudes": 0.01},
}
_INITIAL = np.array([0.0, 0.0])
_MATRIX = np.array([[0.0, 0.0], [0.1, 0.1]])


def _sphere(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    return float(np.sum(variables**2))


def _optimize(**kwargs: Any) -> None:
    optimize(_CONFIG, _INITIAL, _sphere, **kwargs)


def _optimize_many(**kwargs: Any) -> None:
    optimize_many(_CONFIG, _MATRIX, _sphere, **kwargs)


def _evaluate(**kwargs: Any) -> None:
    evaluate(_CONFIG, _INITIAL, _sphere, **kwargs)


def _evaluate_batch(**kwargs: Any) -> None:
    evaluate_batch(_CONFIG, _MATRIX, _sphere, **kwargs)


def _offload(**kwargs: Any) -> None:
    offload(partial(pow, 2, 3), **kwargs)


_TAKES_A_POOL = pytest.mark.parametrize(
    "entry_point",
    [
        pytest.param(_optimize, id="optimize"),
        pytest.param(_optimize_many, id="optimize_many"),
        pytest.param(_evaluate, id="evaluate"),
        pytest.param(_evaluate_batch, id="evaluate_batch"),
        pytest.param(_offload, id="offload"),
    ],
)


@_TAKES_A_POOL
def test_live_pool_accepted(opened: Session, entry_point: Callable[..., None]) -> None:
    entry_point(pool=opened.thread_pool(workers=2))


@_TAKES_A_POOL
def test_pool_from_a_closed_session_refused(entry_point: Callable[..., None]) -> None:
    with session() as closing:
        pool = closing.thread_pool(workers=2)
    with pytest.raises(WorkflowError, match="released when its session closed"):
        entry_point(pool=pool)


def _offload_again(pool: WorkerPool) -> int:
    return offload(partial(pow, 2, 3), pool=pool)


def _optimize_again(
    pool: WorkerPool, variables: NDArray[np.float64], _: EvaluationFunctionContext
) -> float:
    optimize(_CONFIG, _INITIAL, _sphere, pool=pool)
    return float(np.sum(variables**2))


# Both of these would hang rather than fail if the refusal were dropped, so the
# ceiling turns that regression back into a test failure.


@pytest.mark.timeout(30)
def test_offload_to_the_pool_it_runs_on_refused(opened: Session) -> None:
    pool = opened.thread_pool(workers=1)
    with pytest.raises(WorkflowError, match="already running on it"):
        offload(partial(_offload_again, pool), pool=pool)


@pytest.mark.timeout(30)
def test_nested_run_on_the_pool_it_runs_on_refused(opened: Session) -> None:
    pool = opened.thread_pool(workers=1)
    with pytest.raises(WorkflowError, match="already running on it"):
        optimize(_CONFIG, _INITIAL, partial(_optimize_again, pool), pool=pool)


@pytest.mark.timeout(30)
def test_nested_run_on_a_second_pool_allowed(opened: Session) -> None:
    # The control: what makes the refusal above about *this* pool rather
    # than about nesting, which is supported.
    inner = opened.thread_pool(workers=1)
    outer = opened.thread_pool(workers=1)
    optimize(_CONFIG, _INITIAL, partial(_optimize_again, inner), pool=outer)


def _evaluate_with(carried: Any, variables: NDArray[np.float64], _: Any) -> float:
    assert carried is not None
    return float(np.sum(variables**2))


def _pool_of(opened: Session) -> Any:
    return opened.thread_pool(workers=1)


def _handler_of(_opened: Session) -> Any:
    return HistoryHandler()


@pytest.mark.slow
@pytest.mark.parametrize(
    "carry",
    [
        pytest.param(_pool_of, id="pool"),
        pytest.param(_handler_of, id="handler"),
    ],
)
def test_carrying_a_workflow_object_into_a_worker(
    opened: Session, carry: Callable[[Session], Any]
) -> None:
    # An evaluation function that closes over a workflow object cannot be sent:
    # the object holds a lock, so serializing the work item fails.
    function = partial(_evaluate_with, carry(opened))
    with pytest.raises(ExecutionError, match="could not be sent to a worker"):
        optimize(_CONFIG, _INITIAL, function, pool=opened.process_pool(workers=2))
