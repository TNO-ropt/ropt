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
from ropt.simple import session

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


def _waits_then_holds(barrier: threading.Barrier, release: threading.Event) -> Any:
    # Held inside the evaluation until the other run has failed, so this one is
    # provably still going when it reaches the boundary that observes the stop.
    waited = False

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        nonlocal waited
        if not waited:
            waited = True
            barrier.wait(timeout=30)
            assert release.wait(timeout=30)
        return _sphere(variables, context)

    return objective


def _fails_at(barrier: threading.Barrier) -> Any:
    def objective(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        barrier.wait(timeout=30)
        msg = "boom"
        raise ValueError(msg)

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

        result = pool.optimize(_CONFIG, _INITIAL, objective)

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
            results = pool.optimize_many(
                _CONFIG,
                np.tile(_INITIAL, (runs, 1)),
                [_waits_once(started) for _ in range(runs)],
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
            outcome.append(pool.optimize(_CONFIG, _INITIAL, _waits_once(started)))

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
        result = pool.optimize(_CONFIG, _INITIAL, _sphere)

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

        result = other_pool.optimize(_CONFIG, _INITIAL, objective)
        # A stop is not a release: the stopped session's pool still runs.
        assert stopped_pool.optimize(_CONFIG, _INITIAL, _sphere).results is not None

    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


def _run_beside_a_failure(
    *, keep_going: bool | None, failure_keeps_going: bool | None = None
) -> OptimizationResult:
    # One run fails while a second is held inside an evaluation, so the second
    # is certainly still going when the failure lands. Returns the second.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []

    with session() as opened:
        pool = opened.thread_pool(workers=2)

        def _survivor() -> None:
            outcome.append(
                pool.optimize(
                    _CONFIG,
                    _INITIAL,
                    _waits_then_holds(started, release),
                    keep_going=keep_going,
                )
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                pool.optimize(
                    _CONFIG,
                    _INITIAL,
                    _fails_at(started),
                    keep_going=failure_keeps_going,
                )
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    return outcome[0]


@pytest.mark.timeout(60)
def test_a_failing_run_stops_the_others_on_its_session() -> None:
    result = _run_beside_a_failure(keep_going=None)
    assert result.exit_code == ExitCode.FAILED_ELSEWHERE
    assert result.results is not None


@pytest.mark.timeout(60)
def test_keep_going_lets_a_run_finish_when_another_fails() -> None:
    result = _run_beside_a_failure(keep_going=True)
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


@pytest.mark.timeout(60)
def test_a_failing_run_that_keeps_going_still_stops_the_others() -> None:
    # The flag exempts a run from being stopped, never from stopping the rest:
    # a run that may outlive a failure must not be able to hide its own.
    result = _run_beside_a_failure(keep_going=None, failure_keeps_going=True)
    assert result.exit_code == ExitCode.FAILED_ELSEWHERE


@pytest.mark.timeout(60)
def test_a_run_takes_keep_going_from_its_session() -> None:
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []

    with session(keep_going=True) as opened:
        pool = opened.thread_pool(workers=2)

        def _survivor() -> None:
            outcome.append(
                pool.optimize(_CONFIG, _INITIAL, _waits_then_holds(started, release))
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                pool.optimize(_CONFIG, _INITIAL, _fails_at(started))
        finally:
            release.set()
            driver.join(timeout=30)

    assert outcome[0].exit_code == ExitCode.OPTIMIZER_FINISHED


@pytest.mark.timeout(60)
def test_keep_going_on_a_run_overrides_its_session() -> None:
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []

    with session(keep_going=True) as opened:
        pool = opened.thread_pool(workers=2)

        def _survivor() -> None:
            outcome.append(
                pool.optimize(
                    _CONFIG,
                    _INITIAL,
                    _waits_then_holds(started, release),
                    keep_going=False,
                )
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                pool.optimize(_CONFIG, _INITIAL, _fails_at(started))
        finally:
            release.set()
            driver.join(timeout=30)

    assert outcome[0].exit_code == ExitCode.FAILED_ELSEWHERE


@pytest.mark.timeout(60)
def test_session_stop_reaches_a_run_that_keeps_going() -> None:
    # Opting out is about a sibling's failure, not about being asked to stop,
    # and the exit code says which of the two happened.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []

    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def _survivor() -> None:
            outcome.append(
                pool.optimize(
                    _CONFIG,
                    _INITIAL,
                    _waits_then_holds(started, release),
                    keep_going=True,
                )
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        started.wait(timeout=30)
        opened.stop()
        release.set()
        driver.join(timeout=30)

    assert outcome[0].exit_code == ExitCode.CANCELLED
