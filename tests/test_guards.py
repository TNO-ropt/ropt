"""Tests for how a run behaves on a session or pool that cannot serve it."""

# Every case is asserted at every run method the surface has. The methods do
# not share a single code path, so a behaviour that changes for one is easy to
# miss in another, and the parametrization is what makes that visible.
# test_live_pool_accepted and test_run_on_an_open_session_accepted are the
# controls: without them a refusal test would still pass if the method had
# stopped working at all.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt import HistoryHandler, session
from ropt.components.concurrency import AbortSignal
from ropt.exceptions import ExecutionError, WorkflowError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

    from ropt import EvaluationFunctionContext, Session, WorkerPool


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


def _optimize(pool: WorkerPool) -> None:
    pool.optimize(_CONFIG, _INITIAL, _sphere)


def _optimize_many(pool: WorkerPool) -> None:
    pool.optimize_many(_CONFIG, _MATRIX, _sphere)


def _evaluate(pool: WorkerPool) -> None:
    pool.evaluate(_CONFIG, _INITIAL, _sphere)


def _evaluate_batch(pool: WorkerPool) -> None:
    pool.evaluate_batch(_CONFIG, _MATRIX, _sphere)


def _offload(pool: WorkerPool) -> None:
    pool.offload(partial(pow, 2, 3))


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
def test_live_pool_accepted(
    opened: Session, entry_point: Callable[[WorkerPool], None]
) -> None:
    entry_point(opened.thread_pool(workers=2))


@_TAKES_A_POOL
def test_pool_from_a_closed_session_refused(
    entry_point: Callable[[WorkerPool], None],
) -> None:
    with session() as closing:
        pool = closing.thread_pool(workers=2)
    with pytest.raises(WorkflowError, match="released when its session closed"):
        entry_point(pool)


def _session_optimize(opened: Session) -> None:
    opened.optimize(_CONFIG, _INITIAL, _sphere)


def _session_optimize_many(opened: Session) -> None:
    opened.optimize_many(_CONFIG, _MATRIX, _sphere)


def _session_evaluate(opened: Session) -> None:
    opened.evaluate(_CONFIG, _INITIAL, _sphere)


def _session_evaluate_batch(opened: Session) -> None:
    opened.evaluate_batch(_CONFIG, _MATRIX, _sphere)


_RUNS_ON_A_SESSION = pytest.mark.parametrize(
    "entry_point",
    [
        pytest.param(_session_optimize, id="optimize"),
        pytest.param(_session_optimize_many, id="optimize_many"),
        pytest.param(_session_evaluate, id="evaluate"),
        pytest.param(_session_evaluate_batch, id="evaluate_batch"),
    ],
)


@_RUNS_ON_A_SESSION
def test_run_on_an_open_session_accepted(
    opened: Session, entry_point: Callable[[Session], None]
) -> None:
    entry_point(opened)


@_RUNS_ON_A_SESSION
def test_run_on_a_closed_session_refused(
    entry_point: Callable[[Session], None],
) -> None:
    with session() as closing:
        pass
    with pytest.raises(WorkflowError, match="not open"):
        entry_point(closing)


def test_run_starting_while_a_session_closes_refused() -> None:
    # The callback fires from inside the close, which is the one instant at
    # which a run could slip past both the refusal and the abort.
    outcomes: list[str] = []
    closing = session()
    signal = AbortSignal()

    def _start_a_run() -> None:
        try:
            closing.optimize(_CONFIG, _INITIAL, _sphere)
        except WorkflowError:
            outcomes.append("refused")
        else:
            outcomes.append("ran")

    signal.add_callback(_start_a_run)
    with closing:
        closing._register(signal)  # ruff: ignore[private-member-access]
    assert outcomes == ["refused"]


def _offload_again(pool: WorkerPool) -> int:
    return pool.offload(partial(pow, 2, 3))


def _optimize_again(
    pool: WorkerPool, variables: NDArray[np.float64], _: EvaluationFunctionContext
) -> float:
    pool.optimize(_CONFIG, _INITIAL, _sphere)
    return float(np.sum(variables**2))


# Both of these would hang rather than fail if the refusal were dropped, so the
# ceiling turns that regression back into a test failure.


@pytest.mark.timeout(30)
def test_offload_to_the_pool_it_runs_on_refused(opened: Session) -> None:
    pool = opened.thread_pool(workers=1)
    with pytest.raises(WorkflowError, match="already running on it"):
        pool.offload(partial(_offload_again, pool))


@pytest.mark.timeout(30)
def test_nested_run_on_the_pool_it_runs_on_refused(opened: Session) -> None:
    pool = opened.thread_pool(workers=1)
    with pytest.raises(WorkflowError, match="already running on it"):
        pool.optimize(_CONFIG, _INITIAL, partial(_optimize_again, pool))


@pytest.mark.timeout(30)
def test_nested_run_on_a_second_pool_allowed(opened: Session) -> None:
    # The control: what makes the refusal above about *this* pool rather
    # than about nesting, which is supported.
    inner = opened.thread_pool(workers=1)
    outer = opened.thread_pool(workers=1)
    outer.optimize(_CONFIG, _INITIAL, partial(_optimize_again, inner))


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
        opened.process_pool(workers=2).optimize(_CONFIG, _INITIAL, function)
