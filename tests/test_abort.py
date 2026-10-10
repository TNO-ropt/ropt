"""Tests for aborting the runs that are using a session's pools."""

# An abort takes effect at a run's next evaluation boundary, so every test here
# gets its ordering from a barrier or from the evaluation function itself:
# waiting for the effect would only make the assertions likely to hold.

from __future__ import annotations

import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from ropt import session
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt import (
        EvaluationFunctionContext,
        EvaluationResult,
        OptimizationResult,
        Session,
    )
    from ropt.results import FunctionResults

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


def _waits_then_raises(barrier: threading.Barrier, release: threading.Event) -> Any:
    # Raises only once the abort has landed, so what it raises is a consequence
    # of being aborted rather than a failure of its own.
    waited = False

    def objective(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        nonlocal waited
        if not waited:
            waited = True
            barrier.wait(timeout=30)
            assert release.wait(timeout=30)
            msg = "boom"
            raise ValueError(msg)
        return 0.0

    return objective


def _hold_at(barrier: threading.Barrier, release: threading.Event) -> int:
    barrier.wait(timeout=30)
    assert release.wait(timeout=30)
    return 0


def _one() -> int:
    return 1


@pytest.mark.timeout(60)
def test_session_abort_keeps_the_results_of_completed_batches() -> None:
    # Aborting from the report callback lands after a batch has finished, so
    # that batch's results survive and the next one does not run.
    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def _abort_on_first_result(_: FunctionResults) -> None:
            opened.abort()

        result = pool.optimize(
            _CONFIG, _INITIAL, _sphere, report=_abort_on_first_result
        )

    assert result.exit_code == ExitCode.USER_ABORT
    assert result.results is not None


@pytest.mark.timeout(60)
def test_session_abort_aborts_the_batch_in_flight() -> None:
    # Aborted from inside the first evaluation. What the run keeps depends on
    # which batches had finished, but the call count is what shows the rest of
    # the work was dropped rather than run out.
    with session() as opened:
        pool = opened.thread_pool(workers=1)
        calls = 0
        lock = threading.Lock()

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            nonlocal calls
            with lock:
                calls += 1
                first = calls == 1
            if first:
                opened.abort()
            return _sphere(variables, context)

        result = pool.optimize(_CONFIG, _INITIAL, objective)

    assert result.exit_code == ExitCode.USER_ABORT
    # A run that was let alone evaluates the vector and a perturbation per
    # variable, many times over; this one stopped almost immediately.
    with lock:
        assert calls <= _INITIAL.size


@pytest.mark.timeout(60)
def test_session_abort_leaves_an_evaluation_batch_without_results() -> None:
    # One worker and one vector per bundle, so the two rows behind the one that
    # aborts are still queued and are dropped. A batch is all or nothing, so
    # the row that did run is not reported either.
    matrix = np.array([_INITIAL, np.zeros(_INITIAL.size), np.ones(_INITIAL.size)])

    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            opened.abort()
            return _sphere(variables, context)

        outcome = pool.evaluate_batch(_CONFIG, matrix, objective, bundle_size=1)

    assert outcome.exit_code == ExitCode.USER_ABORT
    assert outcome.results == ()


@pytest.mark.timeout(60)
def test_an_in_process_evaluation_batch_is_aborted_between_its_rows() -> None:
    # Without a pool the rows are evaluated one after another on the calling
    # thread, so that loop is the only place the abort can be observed.
    matrix = np.array([_INITIAL, np.zeros(_INITIAL.size), np.ones(_INITIAL.size)])
    calls = 0

    with session() as opened:

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            nonlocal calls
            calls += 1
            opened.abort()
            return _sphere(variables, context)

        outcome = opened.evaluate_batch(_CONFIG, matrix, objective)

    assert outcome.exit_code == ExitCode.USER_ABORT
    assert outcome.results == ()
    assert calls == 1


@pytest.mark.timeout(60)
def test_an_evaluation_that_completes_despite_an_abort_reports_its_results() -> None:
    # The single vector is already on a worker when the abort arrives, and a
    # worker cannot be interrupted, so nothing was lost and nothing is dropped.
    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            opened.abort()
            return _sphere(variables, context)

        outcome = pool.evaluate(_CONFIG, _INITIAL, objective)

    assert outcome.exit_code == ExitCode.FINISHED
    assert outcome.results is not None


@pytest.mark.timeout(60)
def test_a_nested_run_that_fails_does_not_abort_the_run_that_started_it() -> None:
    # The outer run is not a sibling: it receives the exception itself, and
    # aborting it here would arrive first and be all it could report.
    with session() as opened:
        outer = opened.thread_pool(workers=1)
        inner = opened.thread_pool(workers=1)

        def _nested(
            _variables: NDArray[np.float64], _context: EvaluationFunctionContext
        ) -> float:
            msg = "boom"
            raise ValueError(msg)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            inner.optimize(_CONFIG, _INITIAL, _nested)
            return _sphere(variables, context)

        with pytest.raises(ValueError, match="boom"):
            outer.optimize(_CONFIG, _INITIAL, objective)


@pytest.mark.timeout(60)
def test_session_abort_aborts_every_run_in_progress() -> None:
    runs = 3
    started = threading.Barrier(runs + 1)

    with session() as opened:
        pool = opened.thread_pool(workers=runs)

        def abort_once_all_have_started() -> None:
            started.wait(timeout=30)
            opened.abort()

        stopper = threading.Thread(target=abort_once_all_have_started)
        stopper.start()
        try:
            results = pool.optimize_many(
                _CONFIG,
                np.tile(_INITIAL, (runs, 1)),
                [_waits_once(started) for _ in range(runs)],
            )
        finally:
            stopper.join(timeout=30)

    assert [result.exit_code for result in results] == [ExitCode.USER_ABORT] * runs


@pytest.mark.timeout(60)
def test_closing_a_session_aborts_a_run_on_another_thread() -> None:
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
    assert [result.exit_code for result in outcome] == [ExitCode.ABORTED]


@pytest.mark.timeout(60)
def test_a_run_started_after_an_abort_is_unaffected() -> None:
    # The abort is not a latch: it reaches the runs registered at the moment of
    # the call, which is what lets a loop abandon one attempt and start another.
    with session() as opened:
        pool = opened.thread_pool(workers=1)
        opened.abort()
        result = pool.optimize(_CONFIG, _INITIAL, _sphere)

    assert result.exit_code == ExitCode.FINISHED


@pytest.mark.timeout(60)
def test_an_abort_does_not_reach_another_session() -> None:
    with session() as aborted, session() as other:
        aborted_pool = aborted.thread_pool(workers=1)
        other_pool = other.thread_pool(workers=1)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            aborted.abort()
            return _sphere(variables, context)

        result = other_pool.optimize(_CONFIG, _INITIAL, objective)
        # An abort is not a release: the aborted session's pool still runs.
        assert aborted_pool.optimize(_CONFIG, _INITIAL, _sphere).results is not None

    assert result.exit_code == ExitCode.FINISHED


@pytest.mark.timeout(60)
def test_a_failing_run_aborts_the_others_on_its_session() -> None:
    # One run fails while a second is held inside an evaluation, so the second
    # is certainly still going when the failure lands.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []

    with session() as opened:
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

    assert not driver.is_alive()
    assert outcome[0].exit_code == ExitCode.ABORTED_ON_ERROR


@pytest.mark.timeout(60)
def test_session_abort_aborts_the_runs_queued_behind_the_limit() -> None:
    # One at a time, so the four behind the first are still queued when the
    # abort arrives. Each run counts its own evaluations, which is what shows
    # that none of the four reached one.
    lock = threading.Lock()
    calls = [0] * 5

    def _counts(index: int, opened: Session) -> Any:
        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            with lock:
                calls[index] += 1
            if index == 0:
                opened.abort()
            return _sphere(variables, context)

        return objective

    with session() as opened:
        pool = opened.thread_pool(workers=1)
        outcomes = pool.optimize_many(
            _CONFIG,
            np.tile(_INITIAL, (5, 1)),
            [_counts(index, opened) for index in range(5)],
            limit=1,
        )

    assert [outcome.exit_code for outcome in outcomes] == [ExitCode.USER_ABORT] * 5
    with lock:
        assert calls[0] >= 1
        assert calls[1:] == [0, 0, 0, 0]


@pytest.mark.timeout(60)
def test_an_aborted_run_does_not_report_a_failure() -> None:
    # The evaluation raises because its run was aborted. Reporting that as a
    # failure would abort a run that started after the abort and is innocent.
    aborted_started = threading.Barrier(2, timeout=30)
    running = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []
    raised: list[BaseException] = []

    with session() as opened:
        aborted_pool = opened.thread_pool(workers=1)
        fresh_pool = opened.thread_pool(workers=1)

        def _aborted_run() -> None:
            try:
                aborted_pool.optimize(
                    _CONFIG, _INITIAL, _waits_then_raises(aborted_started, release)
                )
            except ValueError as exc:
                raised.append(exc)

        def _fresh_run() -> None:
            outcome.append(fresh_pool.optimize(_CONFIG, _INITIAL, _waits_once(running)))

        aborted_driver = threading.Thread(target=_aborted_run)
        aborted_driver.start()
        aborted_started.wait(timeout=30)
        opened.abort()
        fresh_driver = threading.Thread(target=_fresh_run)
        fresh_driver.start()
        running.wait(timeout=30)
        release.set()
        aborted_driver.join(timeout=30)
        fresh_driver.join(timeout=30)

    assert [str(exc) for exc in raised] == ["boom"]
    assert outcome[0].exit_code == ExitCode.FINISHED


@pytest.mark.timeout(60)
def test_a_failing_run_aborts_an_evaluation_beside_it() -> None:
    # One worker and one vector per bundle, so the second row is still queued
    # when the other run fails; each run has a pool of its own so that the two
    # can overlap on one worker each.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    matrix = np.array([_INITIAL, np.zeros(_INITIAL.size)])
    outcome: list[EvaluationResult[tuple[FunctionResults, ...]]] = []

    with session() as opened:
        evaluating = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)

        def _survivor() -> None:
            outcome.append(
                evaluating.evaluate_batch(
                    _CONFIG, matrix, _waits_then_holds(started, release), bundle_size=1
                )
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                failing.optimize(_CONFIG, _INITIAL, _fails_at(started))
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert outcome[0].exit_code == ExitCode.ABORTED_ON_ERROR
    assert outcome[0].results == ()


@pytest.mark.timeout(60)
def test_an_evaluation_that_cannot_be_built_aborts_the_runs_beside_it() -> None:
    # The failure is in the setup of the evaluation, before any evaluation
    # boundary, so this thread is the barrier's second party rather than it.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    broken = {**_CONFIG, "objectives": {"weights": [0.75, -0.25]}}
    outcome: list[OptimizationResult] = []

    with session() as opened:
        surviving = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)

        def _survivor() -> None:
            outcome.append(
                surviving.optimize(
                    _CONFIG, _INITIAL, _waits_then_holds(started, release)
                )
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            started.wait(timeout=30)
            with pytest.raises(ValidationError, match="Weights must not be negative"):
                failing.evaluate(broken, _INITIAL, _sphere)
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert outcome[0].exit_code == ExitCode.ABORTED_ON_ERROR


@pytest.mark.timeout(60)
def test_an_optimize_many_whose_arguments_disagree_aborts_the_runs_beside_it() -> None:
    # The failure is in the setup of the call, before any run is created, so
    # this thread is the barrier's second party rather than the failing call.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    starts = np.array([_INITIAL, np.zeros(_INITIAL.size)])
    outcome: list[OptimizationResult] = []

    with session() as opened:
        surviving = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)

        def _survivor() -> None:
            outcome.append(
                surviving.optimize(
                    _CONFIG, _INITIAL, _waits_then_holds(started, release)
                )
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            started.wait(timeout=30)
            with pytest.raises(ValueError, match="same length"):
                failing.optimize_many(_CONFIG, starts, [_sphere, _sphere, _sphere])
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert outcome[0].exit_code == ExitCode.ABORTED_ON_ERROR


@pytest.mark.parametrize(
    ("call", "variables", "message"),
    [
        pytest.param("evaluate", np.zeros((2, 3)), "single vector", id="evaluate"),
        pytest.param("evaluate_batch", _INITIAL, "2-D matrix", id="evaluate_batch"),
    ],
)
@pytest.mark.timeout(60)
def test_an_evaluation_of_the_wrong_shape_aborts_the_runs_beside_it(
    call: str, variables: Any, message: str
) -> None:
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []

    with session() as opened:
        surviving = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)

        def _survivor() -> None:
            outcome.append(
                surviving.optimize(
                    _CONFIG, _INITIAL, _waits_then_holds(started, release)
                )
            )

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            started.wait(timeout=30)
            with pytest.raises(ValueError, match=message):
                getattr(failing, call)(_CONFIG, variables, _sphere)
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert outcome[0].exit_code == ExitCode.ABORTED_ON_ERROR


@pytest.mark.timeout(60)
def test_a_failing_run_aborts_an_offload_beside_it() -> None:
    # The second call is still queued behind the single worker when the other
    # run fails, so the abort keeps it from running.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    raised: list[AbortedError] = []

    with session() as opened:
        offloading = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)
        work: list[Callable[[], int]] = [partial(_hold_at, started, release), _one]

        def _survivor() -> None:
            try:
                offloading.offload(work)
            except AbortedError as exc:
                raised.append(exc)

        driver = threading.Thread(target=_survivor)
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                failing.optimize(_CONFIG, _INITIAL, _fails_at(started))
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert [exc.exit_code for exc in raised] == [ExitCode.ABORTED_ON_ERROR]
