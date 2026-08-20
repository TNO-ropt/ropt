"""Tests for the checks the entry points run on pools and handler groups."""

# Every check is asserted at every entry point that takes the argument. The
# entry points do not share a single code path, so a check added to one is
# easy to forget in another, and the parametrization is what makes that
# visible. test_live_pool_accepted is the control: without it a refusal test
# would still pass if the entry point had stopped working for any pool at all.
#
# The transferred cases here pickle by hand rather than go through a worker
# process. A worker is what produces a placeholder in practice, but pickling is
# the whole of the mechanism, and doing it in-process keeps the failure
# readable. test_carrying_a_session_object_into_a_worker covers the real path,
# where the executor's own check reports the object before the run even starts.

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components._transferred import reset_transferred
from ropt.enums import ExitCode
from ropt.exceptions import TransferError, WorkflowError
from ropt.simple import (
    HistoryHandler,
    evaluate,
    evaluate_many,
    offload,
    optimize,
    optimize_many,
    session,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt.simple import EvaluationFunctionContext, Session, WorkerPool

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


def _evaluate_many(**kwargs: Any) -> None:
    evaluate_many(_CONFIG, _MATRIX, _sphere, **kwargs)


def _offload(**kwargs: Any) -> None:
    offload(partial(pow, 2, 3), **kwargs)


_TAKES_A_POOL = pytest.mark.parametrize(
    "entry_point",
    [
        pytest.param(_optimize, id="optimize"),
        pytest.param(_optimize_many, id="optimize_many"),
        pytest.param(_evaluate, id="evaluate"),
        pytest.param(_evaluate_many, id="evaluate_many"),
        pytest.param(_offload, id="offload"),
    ],
)

_TAKES_HANDLERS = pytest.mark.parametrize(
    "entry_point",
    [
        pytest.param(_optimize, id="optimize"),
        pytest.param(_optimize_many, id="optimize_many"),
        pytest.param(_evaluate, id="evaluate"),
        pytest.param(_evaluate_many, id="evaluate_many"),
    ],
)


@_TAKES_A_POOL
def test_live_pool_accepted(entry_point: Callable[..., None]) -> None:
    with session() as active:
        entry_point(pool=active.thread_pool(workers=2))


@_TAKES_A_POOL
def test_closed_pool_refused(entry_point: Callable[..., None]) -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
        pool.close()
        with pytest.raises(WorkflowError, match="closed"):
            entry_point(pool=pool)


@_TAKES_A_POOL
def test_pool_from_a_closed_session_refused(entry_point: Callable[..., None]) -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
    with pytest.raises(WorkflowError, match="closed"):
        entry_point(pool=pool)


@_TAKES_A_POOL
def test_transferred_pool_refused(entry_point: Callable[..., None]) -> None:
    with session() as active:
        pool = _round_trip(active.thread_pool(workers=1))
        with pytest.raises(WorkflowError, match="A worker pool cannot be used"):
            entry_point(pool=pool)


@_TAKES_HANDLERS
def test_closed_group_refused(entry_point: Callable[..., None]) -> None:
    with session() as active:
        group = active.shared_handlers(HistoryHandler())
        group.close()
        with pytest.raises(WorkflowError, match="closed"):
            entry_point(handlers=[group])


@_TAKES_HANDLERS
def test_group_from_a_closed_session_refused(entry_point: Callable[..., None]) -> None:
    with session() as active:
        group = active.shared_handlers(HistoryHandler())
    with pytest.raises(WorkflowError, match="closed"):
        entry_point(handlers=[group])


@_TAKES_HANDLERS
def test_transferred_group_refused(entry_point: Callable[..., None]) -> None:
    with session() as active:
        group = _round_trip(active.shared_handlers(HistoryHandler()))
        with pytest.raises(WorkflowError, match="Shared handlers cannot be used"):
            entry_point(handlers=[group])


def _round_trip(obj: Any) -> Any:
    # What a worker process does to an object it receives, without the worker.
    placeholder = pickle.loads(pickle.dumps(obj))  # ruff: ignore[suspicious-pickle-usage]
    # Unpickling records the transfer in a process-global registry that a real
    # worker would act on; this is the main process, so leave it as it was.
    reset_transferred()
    return placeholder


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
def test_offload_to_the_pool_it_runs_on_refused() -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
        with pytest.raises(WorkflowError, match="already running on it"):
            offload(partial(_offload_again, pool), pool=pool)


@pytest.mark.timeout(30)
def test_nested_run_on_the_pool_it_runs_on_refused() -> None:
    with session() as active:
        pool = active.thread_pool(workers=1)
        with pytest.raises(WorkflowError, match="already running on it"):
            optimize(_CONFIG, _INITIAL, partial(_optimize_again, pool), pool=pool)


@pytest.mark.timeout(30)
def test_nested_run_on_a_second_pool_allowed() -> None:
    # The control: what makes the refusal above about *this* pool rather than
    # about nesting, which is supported.
    with session() as active:
        inner = active.thread_pool(workers=1)
        outer = active.thread_pool(workers=1)
        optimize(_CONFIG, _INITIAL, partial(_optimize_again, inner), pool=outer)


def _close_and_evaluate(
    pool: WorkerPool, variables: NDArray[np.float64], _: EvaluationFunctionContext
) -> float:
    pool.close()
    return float(np.sum(variables**2))


def test_pool_closed_during_a_run_still_stops_the_run() -> None:
    # The checks run once, when the run starts. A pool that dies later is not a
    # misuse of the API but a failure of the workers, and it keeps reporting
    # itself as one: the run ends, it is not refused.
    config = _CONFIG | {"optimizer": {"max_functions": 20}}
    with session() as active:
        pool = active.thread_pool(workers=2)
        result = optimize(
            config, _INITIAL, partial(_close_and_evaluate, pool), pool=pool
        )
    assert result.exit_code == ExitCode.EXECUTOR_STOPPED


def _evaluate_with(carried: Any, variables: NDArray[np.float64], _: Any) -> float:
    assert carried is not None
    return float(np.sum(variables**2))


def _pool_of(active: Session) -> Any:
    return active.thread_pool(workers=1)


def _executor_of(active: Session) -> Any:
    return active.thread_pool(workers=1).executor


def _group_of(active: Session) -> Any:
    return active.shared_handlers(HistoryHandler())


@pytest.mark.slow
@pytest.mark.parametrize(
    ("carry", "subject"),
    [
        pytest.param(_pool_of, "A worker pool", id="pool"),
        pytest.param(_executor_of, "An executor", id="executor"),
        pytest.param(_group_of, "Shared handlers", id="handlers"),
    ],
)
def test_carrying_a_session_object_into_a_worker(
    carry: Callable[[Session], Any], subject: str
) -> None:
    # An evaluation function that closes over a session object drags it into
    # the worker process. The worker reports it by name as soon as it unpacks
    # the work, so the run fails before the function is ever called.
    with session() as active:
        function = partial(_evaluate_with, carry(active))
        with pytest.raises(TransferError, match=subject):
            optimize(_CONFIG, _INITIAL, function, pool=active.process_pool(workers=2))
