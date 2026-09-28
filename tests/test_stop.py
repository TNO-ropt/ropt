"""Tests for stopping the runs that are using a session's pools."""

# A stop takes effect at a run's next evaluation boundary, so every test here
# gets its ordering from a barrier or from the evaluation function itself:
# waiting for the effect would only make the assertions likely to hold.

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.enums import ExitCode
from ropt.simple import optimize, optimize_many, session

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.simple import EvaluationFunctionContext, OptimizationResult

_INITIAL = np.array([0.0, 0.0, 0.1])

_CONFIG: dict[str, Any] = {
    "optimizer": {"max_functions": 20},
    "backend": {"method": "slsqp", "max_iterations": 15},
    "variables": {"variable_count": _INITIAL.size, "perturbation_magnitudes": 0.01},
}


def _sphere(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    return float(np.sum((variables - 0.5) ** 2))


def _waits_once(barrier: threading.Barrier) -> Any:
    # The rendezvous sits inside the run, so passing it proves the run is in
    # progress and not merely about to start.
    waited = False

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        nonlocal waited
        if not waited:
            waited = True
            barrier.wait(timeout=30)
        return _sphere(variables, context)

    return objective


@pytest.mark.timeout(60)
def test_session_stop_cancels_a_run_keeping_its_best_result() -> None:
    with session() as opened:
        pool = opened.thread_pool(workers=1)
        calls = 0

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            nonlocal calls
            calls += 1
            if calls == 1:
                opened.stop()
            return _sphere(variables, context)

        result = optimize(_CONFIG, _INITIAL, objective, pool=pool)

    assert result.exit_code == ExitCode.CANCELLED
    assert result.results is not None


@pytest.mark.timeout(60)
def test_session_stop_cancels_every_run_in_progress() -> None:
    runs = 3
    started = threading.Barrier(runs + 1)

    with session() as opened:
        pool = opened.thread_pool(workers=runs)

        def stop_once_all_have_started() -> None:
            started.wait(timeout=30)
            opened.stop()

        stopper = threading.Thread(target=stop_once_all_have_started)
        stopper.start()
        try:
            results = optimize_many(
                _CONFIG,
                np.tile(_INITIAL, (runs, 1)),
                [_waits_once(started) for _ in range(runs)],
                pool=pool,
            )
        finally:
            stopper.join(timeout=30)

    assert [result.exit_code for result in results] == [ExitCode.CANCELLED] * runs


@pytest.mark.timeout(60)
def test_closing_a_session_cancels_a_run_on_another_thread() -> None:
    started = threading.Barrier(2)
    outcome: list[OptimizationResult] = []

    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def run() -> None:
            outcome.append(optimize(_CONFIG, _INITIAL, _waits_once(started), pool=pool))

        driver = threading.Thread(target=run)
        driver.start()
        # The run is inside its first evaluation, so the block below closes on
        # a run that is genuinely in progress.
        started.wait(timeout=30)

    driver.join(timeout=30)
    assert not driver.is_alive()
    assert [result.exit_code for result in outcome] == [ExitCode.CANCELLED]


@pytest.mark.timeout(60)
def test_a_run_started_after_a_stop_is_unaffected() -> None:
    # The stop is not a latch: it reaches the runs registered at the moment of
    # the call, which is what lets a loop stop one attempt and start another.
    with session() as opened:
        pool = opened.thread_pool(workers=1)
        opened.stop()
        result = optimize(_CONFIG, _INITIAL, _sphere, pool=pool)

    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


@pytest.mark.timeout(60)
def test_a_stop_does_not_reach_another_session() -> None:
    with session() as stopped, session() as other:
        stopped_pool = stopped.thread_pool(workers=1)
        other_pool = other.thread_pool(workers=1)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            stopped.stop()
            return _sphere(variables, context)

        result = optimize(_CONFIG, _INITIAL, objective, pool=other_pool)
        assert stopped_pool.executor is not None

    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED
