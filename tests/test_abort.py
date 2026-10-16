"""Tests for aborting the runs that are using a session's pools."""

# An aborted run raises `AbortedError`: at once on a pool, and at its next
# evaluation boundary without one. Every test here gets its ordering from a
# barrier or from the evaluation function itself: waiting for the effect would
# only make the assertions likely to hold.

from __future__ import annotations

import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from ropt import aborting, session
from ropt.components.concurrency import _grace, parent_signal
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt import (
        EvaluationFunctionContext,
        OptimizationResult,
        Session,
    )
    from ropt.components.concurrency import AbortSignal
    from ropt.results import FunctionResults

_INITIAL = np.array([0.0, 0.0, 0.1])

_CONFIG: dict[str, Any] = {
    "optimizer": {"max_functions": 20},
    "backend": {"method": "slsqp", "max_iterations": 15},
    "variables": {"variable_count": _INITIAL.size, "perturbation_magnitudes": 0.01},
}


def _sphere(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    return float(np.sum((variables - 0.5) ** 2))


def _records_abort(run: Callable[[], object], codes: list[ExitCode]) -> None:
    # An aborted run raises; one that finishes records nothing.
    try:
        run()
    except AbortedError as exc:
        codes.append(exc.exit_code)


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


def _records_its_signal_then_raises(
    seen: list[AbortSignal], error: type[BaseException]
) -> Any:
    # Work started from the evaluation function takes this signal as its parent,
    # so whether it aborts is what reaches that work.
    def objective(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        signal = parent_signal()
        assert signal is not None
        seen.append(signal)
        raise error

    return objective


def test_aborting_is_false_outside_any_task() -> None:
    assert not aborting()


@pytest.mark.timeout(60)
def test_aborting_turns_true_in_an_evaluation_function_once_its_run_is_aborted() -> (
    None
):
    seen: list[tuple[bool, bool]] = []

    with session() as opened:

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            before = aborting()
            opened.abort()
            seen.append((before, aborting()))
            return _sphere(variables, context)

        with pytest.raises(AbortedError):
            opened.optimize(_CONFIG, _INITIAL, objective)

    assert seen[0] == (False, True)


@pytest.mark.timeout(60)
def test_aborting_turns_true_in_a_function_given_to_run_once_the_call_is_aborted() -> (
    None
):
    # The `run` call does not wait for the function, so the function signals
    # when it has looked.
    seen: list[tuple[bool, bool]] = []
    looked = threading.Event()

    with session() as opened:

        def _work() -> int:
            before = aborting()
            opened.abort()
            seen.append((before, aborting()))
            looked.set()
            return 0

        with pytest.raises(AbortedError):
            opened.run(_work)
        assert looked.wait(timeout=30)

    assert seen == [(False, True)]


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_optimization_ended_by_a_base_exception_aborts_the_work_it_started(
    error: type[BaseException],
) -> None:
    seen: list[AbortSignal] = []
    with session() as opened:
        with pytest.raises(error):
            opened.optimize(
                _CONFIG, _INITIAL, _records_its_signal_then_raises(seen, error)
            )
        assert seen[0].aborting
        assert seen[0].exit_code is ExitCode.ABORTED


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_evaluation_ended_by_a_base_exception_aborts_the_work_it_started(
    error: type[BaseException],
) -> None:
    seen: list[AbortSignal] = []
    with session() as opened:
        with pytest.raises(error):
            opened.evaluate(
                _CONFIG, _INITIAL, _records_its_signal_then_raises(seen, error)
            )
        assert seen[0].aborting
        assert seen[0].exit_code is ExitCode.ABORTED


def _hold_at(barrier: threading.Barrier, release: threading.Event) -> int:
    barrier.wait(timeout=30)
    assert release.wait(timeout=30)
    return 0


def _one() -> int:
    return 1


@pytest.mark.timeout(60)
def test_session_abort_between_batches_raises() -> None:
    # Aborting from the report callback lands after a batch has finished, and
    # the run raises rather than returning what it had reached.
    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def _abort_on_first_result(_: FunctionResults) -> None:
            opened.abort()

        with pytest.raises(AbortedError) as raised:
            pool.optimize(_CONFIG, _INITIAL, _sphere, report=_abort_on_first_result)

    assert raised.value.exit_code is ExitCode.USER_ABORT


@pytest.mark.timeout(60)
def test_session_abort_aborts_the_batch_in_flight() -> None:
    # Aborted from inside the first evaluation. The call count is what shows the
    # rest of the work was dropped rather than run out.
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

        with pytest.raises(AbortedError) as raised:
            pool.optimize(_CONFIG, _INITIAL, objective)

    assert raised.value.exit_code is ExitCode.USER_ABORT
    # A run that was let alone evaluates the vector and a perturbation per
    # variable, many times over; this one stopped almost immediately.
    with lock:
        assert calls <= _INITIAL.size


@pytest.mark.timeout(60)
def test_session_abort_of_an_evaluation_batch_raises() -> None:
    # One worker and one task per vector, so the two rows behind the one that
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

        with pytest.raises(AbortedError) as raised:
            pool.evaluate_batch(_CONFIG, matrix, objective)

    assert raised.value.exit_code is ExitCode.USER_ABORT


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

        with pytest.raises(AbortedError) as raised:
            opened.evaluate_batch(_CONFIG, matrix, objective)

    assert raised.value.exit_code is ExitCode.USER_ABORT
    assert calls == 1


@pytest.mark.timeout(60)
def test_an_evaluation_aborted_while_its_vector_runs_raises() -> None:
    # The single vector is already on a worker when the abort arrives. A worker
    # cannot be interrupted, and the evaluation does not wait for it.
    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            opened.abort()
            return _sphere(variables, context)

        with pytest.raises(AbortedError) as raised:
            pool.evaluate(_CONFIG, _INITIAL, objective)

    assert raised.value.exit_code is ExitCode.USER_ABORT


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
    codes: list[ExitCode] = []

    with session() as opened:
        pool = opened.thread_pool(workers=runs)
        run = partial(pool.optimize, _CONFIG, _INITIAL)
        drivers = [
            threading.Thread(
                target=_records_abort, args=(partial(run, _waits_once(started)), codes)
            )
            for _ in range(runs)
        ]
        for driver in drivers:
            driver.start()
        try:
            started.wait(timeout=30)
            opened.abort()
        finally:
            for driver in drivers:
                driver.join(timeout=30)

    assert codes == [ExitCode.USER_ABORT] * runs


@pytest.mark.timeout(60)
def test_closing_a_session_aborts_a_run_on_another_thread() -> None:
    started = threading.Barrier(2)
    codes: list[ExitCode] = []

    with session() as opened:
        pool = opened.thread_pool(workers=1)
        run = partial(pool.optimize, _CONFIG, _INITIAL, _waits_once(started))
        driver = threading.Thread(target=_records_abort, args=(run, codes))
        driver.start()
        # The run is inside its first evaluation, so the block below closes on
        # a run that is genuinely in progress.
        started.wait(timeout=30)

    driver.join(timeout=30)
    assert not driver.is_alive()
    assert codes == [ExitCode.ABORTED]


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
    codes: list[ExitCode] = []

    with session() as opened:
        pool = opened.thread_pool(workers=2)
        run = partial(
            pool.optimize, _CONFIG, _INITIAL, _waits_then_holds(started, release)
        )
        driver = threading.Thread(target=_records_abort, args=(run, codes))
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                pool.optimize(_CONFIG, _INITIAL, _fails_at(started))
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert codes == [ExitCode.ABORTED_ON_ERROR]


@pytest.mark.timeout(60)
def test_session_abort_starts_none_of_the_runs_queued_behind_the_limit() -> None:
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
        with pytest.raises(AbortedError) as raised:
            opened.run(
                [
                    partial(pool.optimize, _CONFIG, _INITIAL, _counts(index, opened))
                    for index in range(5)
                ],
                limit=1,
            )

    assert raised.value.exit_code is ExitCode.USER_ABORT
    with lock:
        assert calls[0] >= 1
        assert calls[1:] == [0, 0, 0, 0]


@pytest.mark.timeout(60)
def test_an_aborted_run_does_not_report_a_failure() -> None:
    # The evaluation raises because its run was aborted. Reporting that as a
    # failure would abort a run that started after the abort and is innocent.
    # The aborted run has no pool, so it waits for its evaluation and raises
    # what the evaluation raised.
    aborted_started = threading.Barrier(2, timeout=30)
    running = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []
    raised: list[BaseException] = []

    with session() as opened:
        fresh_pool = opened.thread_pool(workers=1)

        def _aborted_run() -> None:
            try:
                opened.optimize(
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
    # One worker and one task per vector, so the second row is still queued
    # when the other run fails; each run has a pool of its own so that the two
    # can overlap on one worker each.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    matrix = np.array([_INITIAL, np.zeros(_INITIAL.size)])
    codes: list[ExitCode] = []

    with session() as opened:
        evaluating = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)
        run = partial(
            evaluating.evaluate_batch,
            _CONFIG,
            matrix,
            _waits_then_holds(started, release),
        )
        driver = threading.Thread(target=_records_abort, args=(run, codes))
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                failing.optimize(_CONFIG, _INITIAL, _fails_at(started))
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert codes == [ExitCode.ABORTED_ON_ERROR]


@pytest.mark.timeout(60)
def test_an_evaluation_that_cannot_be_built_aborts_the_runs_beside_it() -> None:
    # The failure is in the setup of the evaluation, before any evaluation
    # boundary, so this thread is the barrier's second party rather than it.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    broken = {**_CONFIG, "objectives": {"weights": [0.75, -0.25]}}
    codes: list[ExitCode] = []

    with session() as opened:
        surviving = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)
        run = partial(
            surviving.optimize, _CONFIG, _INITIAL, _waits_then_holds(started, release)
        )
        driver = threading.Thread(target=_records_abort, args=(run, codes))
        driver.start()
        try:
            started.wait(timeout=30)
            with pytest.raises(ValidationError, match="Weights must not be negative"):
                failing.evaluate(broken, _INITIAL, _sphere)
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert codes == [ExitCode.ABORTED_ON_ERROR]


@pytest.mark.timeout(60)
def test_a_session_run_with_an_invalid_limit_aborts_the_tasks_beside_it() -> None:
    # The failure is in the arguments of the call, before any task is created,
    # so this thread is the barrier's second party rather than the failing call.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    codes: list[ExitCode] = []

    with session() as opened:
        surviving = opened.thread_pool(workers=1)
        run = partial(
            surviving.optimize, _CONFIG, _INITIAL, _waits_then_holds(started, release)
        )
        driver = threading.Thread(target=_records_abort, args=(run, codes))
        driver.start()
        try:
            started.wait(timeout=30)
            with pytest.raises(ValueError, match="at least one"):
                opened.run(partial(int, 1), limit=0)
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert codes == [ExitCode.ABORTED_ON_ERROR]


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
    codes: list[ExitCode] = []

    with session() as opened:
        surviving = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)
        run = partial(
            surviving.optimize, _CONFIG, _INITIAL, _waits_then_holds(started, release)
        )
        driver = threading.Thread(target=_records_abort, args=(run, codes))
        driver.start()
        try:
            started.wait(timeout=30)
            with pytest.raises(ValueError, match=message):
                getattr(failing, call)(_CONFIG, variables, _sphere)
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert codes == [ExitCode.ABORTED_ON_ERROR]


@pytest.mark.timeout(60)
def test_a_failing_optimization_aborts_a_pool_run_beside_it() -> None:
    # The second call is still queued behind the single worker when the
    # optimization fails, so the abort keeps it from running.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    raised: list[AbortedError] = []

    with session() as opened:
        running = opened.thread_pool(workers=1)
        failing = opened.thread_pool(workers=1)
        work: list[Callable[[], int]] = [partial(_hold_at, started, release), _one]

        def _survivor() -> None:
            try:
                running.run(work)
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


def _start_on(opened: Session, where: str, work: Callable[[], object]) -> Any:
    if where == "thread_pool":
        return opened.thread_pool(workers=1).run(work)
    return opened.run(work)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("where", ["thread_pool", "session_run"])
def test_session_close_waits_for_aborted_work_to_end(where: str) -> None:
    # The work ends only once it sees the abort, which closing the session
    # sends, so it can only have finished if closing waited for it.
    started = threading.Event()
    finished = threading.Event()

    def work() -> None:
        signal = parent_signal()
        assert signal is not None
        aborted = threading.Event()
        signal.add_callback(aborted.set)
        started.set()
        assert aborted.wait(timeout=30)
        finished.set()

    with session() as opened:
        driver = threading.Thread(
            target=_records_abort, args=(partial(_start_on, opened, where, work), [])
        )
        driver.start()
        assert started.wait(timeout=30)

    assert finished.is_set()
    driver.join(timeout=30)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("where", ["thread_pool", "session_run"])
def test_session_close_stops_waiting_after_the_grace_period(
    where: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Work that ignores the abort is waited for until the grace period ends,
    # and then left running.
    monkeypatch.setattr(_grace, "GRACE_PERIOD", 0.2)
    started = threading.Event()
    release = threading.Event()

    def work() -> None:
        started.set()
        assert release.wait(timeout=30)

    try:
        with session() as opened:
            driver = threading.Thread(
                target=_records_abort,
                args=(partial(_start_on, opened, where, work), []),
            )
            driver.start()
            assert started.wait(timeout=30)
            closing = time.monotonic()
        assert time.monotonic() - closing >= 0.2
        assert not release.is_set()
    finally:
        release.set()
        driver.join(timeout=30)
