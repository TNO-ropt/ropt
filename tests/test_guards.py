"""Tests for how the entry points behave with an executor that cannot serve them."""

# Every case is asserted at every entry point that takes `executor=`. The entry
# points do not share a single code path, so a behaviour that changes for one is
# easy to miss in another, and the parametrization is what makes that visible.
# test_live_executor_accepted is the control: without it a refusal test would
# still pass if the entry point had stopped working for any executor at all.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.exceptions import ExecutionError, WorkflowError
from ropt.simple import (
    HistoryHandler,
    ProcessExecutor,
    ThreadExecutor,
    evaluate,
    evaluate_many,
    offload,
    optimize,
    optimize_many,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt.components.executors import Executor
    from ropt.simple import EvaluationFunctionContext

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


_TAKES_AN_EXECUTOR = pytest.mark.parametrize(
    "entry_point",
    [
        pytest.param(_optimize, id="optimize"),
        pytest.param(_optimize_many, id="optimize_many"),
        pytest.param(_evaluate, id="evaluate"),
        pytest.param(_evaluate_many, id="evaluate_many"),
        pytest.param(_offload, id="offload"),
    ],
)


@_TAKES_AN_EXECUTOR
def test_live_executor_accepted(entry_point: Callable[..., None]) -> None:
    entry_point(executor=ThreadExecutor(workers=2))


def _offload_again(executor: Executor) -> int:
    return offload(partial(pow, 2, 3), executor=executor)


def _optimize_again(
    executor: Executor, variables: NDArray[np.float64], _: EvaluationFunctionContext
) -> float:
    optimize(_CONFIG, _INITIAL, _sphere, executor=executor)
    return float(np.sum(variables**2))


# Both of these would hang rather than fail if the refusal were dropped, so the
# ceiling turns that regression back into a test failure.


@pytest.mark.timeout(30)
def test_offload_to_the_executor_it_runs_on_refused() -> None:
    executor = ThreadExecutor(workers=1)
    with pytest.raises(WorkflowError, match="already running on it"):
        offload(partial(_offload_again, executor), executor=executor)


@pytest.mark.timeout(30)
def test_nested_run_on_the_executor_it_runs_on_refused() -> None:
    executor = ThreadExecutor(workers=1)
    with pytest.raises(WorkflowError, match="already running on it"):
        optimize(
            _CONFIG, _INITIAL, partial(_optimize_again, executor), executor=executor
        )


@pytest.mark.timeout(30)
def test_nested_run_on_a_second_executor_allowed() -> None:
    # The control: what makes the refusal above about *this* executor rather
    # than about nesting, which is supported.
    inner = ThreadExecutor(workers=1)
    outer = ThreadExecutor(workers=1)
    optimize(_CONFIG, _INITIAL, partial(_optimize_again, inner), executor=outer)


def _evaluate_with(carried: Any, variables: NDArray[np.float64], _: Any) -> float:
    assert carried is not None
    return float(np.sum(variables**2))


def _executor_of() -> Any:
    return ThreadExecutor(workers=1)


def _handler_of() -> Any:
    return HistoryHandler()


@pytest.mark.slow
@pytest.mark.parametrize(
    "carry",
    [
        pytest.param(_executor_of, id="executor"),
        pytest.param(_handler_of, id="handler"),
    ],
)
def test_carrying_a_workflow_object_into_a_worker(carry: Callable[[], Any]) -> None:
    # An evaluation function that closes over a workflow object cannot be sent:
    # the object holds a lock, so serializing the work item fails.
    function = partial(_evaluate_with, carry())
    executor = ProcessExecutor(workers=2)
    with pytest.raises(ExecutionError, match="could not be sent to a worker"):
        optimize(_CONFIG, _INITIAL, function, executor=executor)
