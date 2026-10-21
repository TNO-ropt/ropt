"""Tests for aborting the runs that are using a session's pools."""

# An aborted run raises `AbortedError`: at once on a pool, and at its next
# evaluation boundary without one. Every test here gets its ordering from a
# barrier or from the evaluation function itself: waiting for the effect would
# only make the assertions likely to hold.

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from functools import partial
from multiprocessing.connection import Client, Listener
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt import (
    aborting,
    evaluate,
    evaluate_batch,
    optimize,
    run,
    run_program,
    session,
)
from ropt._concurrency import _grace, current_abort_scope
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from ropt import (
        EvaluationFunctionContext,
        OptimizationResult,
        Session,
    )
    from ropt._concurrency import AbortScope
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


def _records_its_scope_then_raises(
    seen: list[AbortScope], error: type[BaseException]
) -> Any:
    # Work started from the evaluation function takes this scope as its parent,
    # so whether it aborts is what reaches that work.
    def objective(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        scope = current_abort_scope()
        assert scope is not None
        seen.append(scope)
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
            optimize(_CONFIG, _INITIAL, objective)

    assert seen[0] == (False, True)


@pytest.mark.timeout(60)
def test_aborting_turns_true_in_a_function_given_to_run_once_the_call_is_aborted() -> (
    None
):
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
            run(_work)
        assert looked.wait(timeout=30)

    assert seen == [(False, True)]


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_optimization_ended_by_a_base_exception_aborts_the_work_it_started(
    error: type[BaseException],
) -> None:
    seen: list[AbortScope] = []
    with session():
        with pytest.raises(error):
            optimize(_CONFIG, _INITIAL, _records_its_scope_then_raises(seen, error))
        assert seen[0].aborting
        assert seen[0].exit_code is ExitCode.ABORTED


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_evaluation_ended_by_a_base_exception_aborts_the_work_it_started(
    error: type[BaseException],
) -> None:
    seen: list[AbortScope] = []
    with session():
        with pytest.raises(error):
            evaluate(_CONFIG, _INITIAL, _records_its_scope_then_raises(seen, error))
        assert seen[0].aborting
        assert seen[0].exit_code is ExitCode.ABORTED


@pytest.mark.timeout(60)
def test_session_abort_between_batches_raises() -> None:
    # Aborting from the report callback lands after a batch has finished, and
    # the run raises rather than returning what it had reached.
    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def _abort_on_first_result(_: FunctionResults) -> None:
            opened.abort()

        with pytest.raises(AbortedError) as raised:
            optimize(
                _CONFIG, _INITIAL, _sphere, pool=pool, report=_abort_on_first_result
            )

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
            optimize(_CONFIG, _INITIAL, objective, pool=pool)

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
            evaluate_batch(_CONFIG, matrix, objective, pool=pool)

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
            evaluate_batch(_CONFIG, matrix, objective)

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
            evaluate(_CONFIG, _INITIAL, objective, pool=pool)

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
            optimize(_CONFIG, _INITIAL, _nested, pool=inner)
            return _sphere(variables, context)

        with pytest.raises(ValueError, match="boom"):
            optimize(_CONFIG, _INITIAL, objective, pool=outer)


@pytest.mark.timeout(60)
def test_session_abort_aborts_every_run_in_progress() -> None:
    runs = 3
    started = threading.Barrier(runs + 1)
    codes: list[ExitCode] = []

    with session() as opened:
        pool = opened.thread_pool(workers=runs)
        start = partial(optimize, _CONFIG, _INITIAL, pool=pool)
        drivers = [
            threading.Thread(
                target=_records_abort,
                args=(partial(start, _waits_once(started)), codes),
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
        start = partial(optimize, _CONFIG, _INITIAL, _waits_once(started), pool=pool)
        driver = threading.Thread(target=_records_abort, args=(start, codes))
        driver.start()
        # The run is inside its first evaluation, so the block below closes on
        # a run that is genuinely in progress.
        started.wait(timeout=30)

    driver.join(timeout=30)
    assert not driver.is_alive()
    assert codes == [ExitCode.ABORTED]


@pytest.mark.parametrize("where", ["module", "pool"])
@pytest.mark.timeout(60)
def test_session_abort_aborts_a_run_started_afterwards(where: str) -> None:
    # The session stays aborted, so a later run aborts before its first
    # evaluation.
    calls: list[int] = []

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        calls.append(1)
        return _sphere(variables, context)

    with session() as opened:
        pool = opened.thread_pool(workers=1)
        opened.abort()
        with pytest.raises(AbortedError) as raised:
            optimize(
                _CONFIG, _INITIAL, objective, pool=None if where == "module" else pool
            )

    assert raised.value.exit_code is ExitCode.USER_ABORT
    assert calls == []


@pytest.mark.parametrize("call", [optimize, evaluate, evaluate_batch])
@pytest.mark.timeout(60)
def test_session_abort_reaches_a_module_function_in_its_block(
    call: Callable[..., object],
) -> None:
    seen: list[bool] = []

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        opened.abort()
        seen.append(aborting())
        return _sphere(variables, context)

    variables = _INITIAL if call is not evaluate_batch else np.array([_INITIAL])
    with session() as opened, contextlib.suppress(AbortedError):
        call(_CONFIG, variables, objective)

    assert seen == [True]


@pytest.mark.timeout(60)
def test_session_abort_reaches_a_session_opened_in_its_block() -> None:
    with session() as outer, session() as inner:
        outer.abort()
        with pytest.raises(AbortedError) as raised:
            optimize(_CONFIG, _INITIAL, _sphere, pool=inner.thread_pool(workers=1))

    assert raised.value.exit_code is ExitCode.USER_ABORT


@pytest.mark.timeout(60)
def test_session_abort_does_not_reach_the_session_it_was_opened_in() -> None:
    with session():
        with session() as inner:
            inner.abort()
        result = optimize(_CONFIG, _INITIAL, _sphere)

    assert result.exit_code == ExitCode.FINISHED


@pytest.mark.timeout(60)
def test_failing_run_does_not_abort_the_others_on_its_session() -> None:
    # One run fails while a second is held inside an evaluation, so the second
    # is certainly still going when the failure is raised.
    started = threading.Barrier(2, timeout=30)
    release = threading.Event()
    outcome: list[OptimizationResult] = []

    with session() as opened:
        pool = opened.thread_pool(workers=2)

        def _held_run() -> None:
            outcome.append(
                optimize(
                    _CONFIG, _INITIAL, _waits_then_holds(started, release), pool=pool
                )
            )

        driver = threading.Thread(target=_held_run)
        driver.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                optimize(_CONFIG, _INITIAL, _fails_at(started), pool=pool)
        finally:
            release.set()
            driver.join(timeout=30)

    assert not driver.is_alive()
    assert outcome[0].exit_code == ExitCode.FINISHED


@pytest.mark.timeout(60)
def test_session_abort_starts_none_of_the_runs_queued_on_a_pool() -> None:
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
            run(
                [
                    partial(
                        optimize, _CONFIG, _INITIAL, _counts(index, opened), pool=pool
                    )
                    for index in range(5)
                ],
                pool=opened.thread_pool(workers=1),
            )

    assert raised.value.exit_code is ExitCode.USER_ABORT
    with lock:
        assert calls[0] >= 1
        assert calls[1:] == [0, 0, 0, 0]


@pytest.mark.timeout(60)
def test_session_close_waits_for_aborted_work_to_end() -> None:
    # The work ends only once it sees the abort, which closing the session
    # sends, so it can only have finished if closing waited for it.
    started = threading.Event()
    finished = threading.Event()

    def work() -> None:
        scope = current_abort_scope()
        assert scope is not None
        aborted = threading.Event()
        scope.add_callback(aborted.set)
        started.set()
        assert aborted.wait(timeout=30)
        finished.set()

    with session() as opened:
        driver = threading.Thread(
            target=_records_abort,
            args=(partial(run, work, pool=opened.thread_pool(workers=1)), []),
        )
        driver.start()
        assert started.wait(timeout=30)

    assert finished.is_set()
    driver.join(timeout=30)


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_session_close_stops_waiting_after_the_grace_period(
    monkeypatch: pytest.MonkeyPatch,
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
                args=(partial(run, work, pool=opened.thread_pool(workers=1)), []),
            )
            driver.start()
            assert started.wait(timeout=30)
            closing = time.monotonic()
        assert time.monotonic() - closing >= 0.2
        assert not release.is_set()
    finally:
        release.set()
        driver.join(timeout=30)


def _report_abort(address: str) -> int:
    # Runs in a worker process, where only an address can be passed in.
    connection = Client(address)
    scope = current_abort_scope()
    assert scope is not None
    aborted = threading.Event()
    scope.add_callback(aborted.set)
    connection.send("started")
    if aborted.wait(timeout=30):
        connection.send(int(scope.exit_code))
    return 0


def _ignore_abort(address: str) -> int:
    # Blocks until the test closes the connection, which never happens while
    # the test is waiting for this process to be killed.
    connection = Client(address)
    connection.send("started")
    connection.recv()
    return 0


def _run_into_fifo(fifo: str) -> int:
    return run_program(["sh", "-c", "echo started; exec sleep 60"], stdout=fifo)


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_session_abort_reaches_work_running_in_a_process_pool(tmp_path: Path) -> None:
    codes: list[ExitCode] = []
    with Listener(str(tmp_path / "socket")) as listener, session() as opened:
        pool = opened.process_pool(workers=1)
        driver = threading.Thread(
            target=_records_abort,
            args=(
                partial(run, partial(_report_abort, listener.address), pool=pool),
                codes,
            ),
        )
        driver.start()
        with listener.accept() as connection:
            assert connection.recv() == "started"
            opened.abort()
            assert connection.poll(30)
            assert connection.recv() == ExitCode.USER_ABORT
        driver.join(timeout=30)
    assert codes == [ExitCode.USER_ABORT]


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_session_abort_kills_busy_process_workers_after_the_grace_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The work ignores the abort, so only killing its worker ends it.
    monkeypatch.setattr(_grace, "GRACE_PERIOD", 0.2)
    codes: list[ExitCode] = []
    with Listener(str(tmp_path / "socket")) as listener, session() as opened:
        pool = opened.process_pool(workers=1)
        driver = threading.Thread(
            target=_records_abort,
            args=(
                partial(run, partial(_ignore_abort, listener.address), pool=pool),
                codes,
            ),
        )
        driver.start()
        with listener.accept() as connection:
            assert connection.recv() == "started"
            opened.abort()
            assert connection.poll(30)
            with pytest.raises(EOFError):
                connection.recv()
        driver.join(timeout=30)
        assert codes == [ExitCode.USER_ABORT]


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_session_close_kills_busy_process_workers_after_the_grace_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_grace, "GRACE_PERIOD", 0.2)
    with Listener(str(tmp_path / "socket")) as listener:
        with session() as opened:
            pool = opened.process_pool(workers=1)
            driver = threading.Thread(
                target=_records_abort,
                args=(
                    partial(run, partial(_ignore_abort, listener.address), pool=pool),
                    [],
                ),
            )
            driver.start()
            connection = listener.accept()
            assert connection.recv() == "started"
        with connection:
            assert connection.poll(30)
            with pytest.raises(EOFError):
                connection.recv()
        driver.join(timeout=30)


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="needs POSIX")
@pytest.mark.timeout(60)
def test_session_abort_kills_a_program_run_in_a_process_pool(tmp_path: Path) -> None:
    # Whoever holds the FIFO open for writing keeps it from reaching end of
    # file, so reaching it shows that the program has exited.
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    codes: list[ExitCode] = []
    with session() as opened:
        pool = opened.process_pool(workers=1)
        driver = threading.Thread(
            target=_records_abort,
            args=(partial(run, partial(_run_into_fifo, str(fifo)), pool=pool), codes),
        )
        driver.start()
        with fifo.open("rb") as stream:
            assert stream.readline() == b"started\n"
            opened.abort()
            assert stream.read() == b""
        driver.join(timeout=30)
    assert codes == [ExitCode.USER_ABORT]


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX")
@pytest.mark.timeout(60)
def test_session_abort_kills_a_program_run_in_a_thread_pool(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    codes: list[ExitCode] = []
    with session() as opened:
        driver = threading.Thread(
            target=_records_abort,
            args=(
                partial(
                    run,
                    partial(_run_into_fifo, str(fifo)),
                    pool=opened.thread_pool(workers=1),
                ),
                codes,
            ),
        )
        driver.start()
        with fifo.open("rb") as stream:
            assert stream.readline() == b"started\n"
            opened.abort()
            assert stream.read() == b""
        driver.join(timeout=30)
    assert codes == [ExitCode.USER_ABORT]


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="needs POSIX")
@pytest.mark.timeout(60)
def test_session_abort_kills_a_program_run_in_a_local_job(tmp_path: Path) -> None:
    # No abort scope reaches a job: ending the job process is what ends the
    # program it started.
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    codes: list[ExitCode] = []
    with session() as opened:
        pool = opened.hpc_pool(workdir=tmp_path, queue="local")
        driver = threading.Thread(
            target=_records_abort,
            args=(partial(run, partial(_run_into_fifo, str(fifo)), pool=pool), codes),
        )
        driver.start()
        with fifo.open("rb") as stream:
            assert stream.readline() == b"started\n"
            opened.abort()
            assert stream.read() == b""
        driver.join(timeout=30)
    assert codes == [ExitCode.USER_ABORT]


def _interrupt_own_process() -> int:
    os.kill(os.getpid(), signal.SIGINT)
    return 1


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="needs POSIX")
@pytest.mark.timeout(60)
def test_process_workers_ignore_keyboard_interrupt() -> None:
    # Ctrl-C at a terminal reaches every process in the foreground group, and
    # handling it is up to the main process.
    with session() as opened:
        assert run(_interrupt_own_process, pool=opened.process_pool(workers=1)) == 1
