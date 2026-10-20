"""Tests for tasks started from inside another task."""

# A task started from code that another one executes is nested in it.
# Its failure is raised to that code, which may handle it; only when the
# exception escapes does the outer one fail and abort the other tasks of the
# session.

from __future__ import annotations

import contextlib
import contextvars
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt import evaluate, evaluate_batch, optimize, run, session
from ropt.components.concurrency import current_abort_scope
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt import (
        EvaluationFunctionContext,
        Session,
        WorkerPool,
    )
    from ropt.results import FunctionResults

_INITIAL = np.array([0.0, 0.0, 0.1])

_CONFIG: dict[str, Any] = {
    "optimizer": {"max_functions": 20},
    "backend": {"method": "slsqp", "max_iterations": 15},
    "variables": {"variable_count": _INITIAL.size, "perturbation_magnitudes": 0.01},
}

_INVALID_CONFIG: dict[str, Any] = {**_CONFIG, "objectives": {"weights": [-1.0]}}


def _sphere(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    return float(np.sum((variables - 0.5) ** 2))


def _boom(*_args: object) -> float:
    msg = "boom"
    raise ValueError(msg)


def _pool_run(_opened: Session, inner: WorkerPool) -> None:
    run(_boom, pool=inner)


def _optimize(_opened: Session, _inner: WorkerPool) -> None:
    optimize(_CONFIG, _INITIAL, _boom)


def _optimize_invalid_config(_opened: Session, _inner: WorkerPool) -> None:
    optimize(_INVALID_CONFIG, _INITIAL, _sphere)


def _evaluate(_opened: Session, _inner: WorkerPool) -> None:
    evaluate(_CONFIG, _INITIAL, _boom)


def _evaluate_invalid_config(_opened: Session, _inner: WorkerPool) -> None:
    evaluate(_INVALID_CONFIG, _INITIAL, _sphere)


def _evaluate_wrong_shape(_opened: Session, _inner: WorkerPool) -> None:
    evaluate(_CONFIG, np.tile(_INITIAL, (2, 1)), _sphere)


def _evaluate_batch(_opened: Session, _inner: WorkerPool) -> None:
    evaluate_batch(_CONFIG, np.tile(_INITIAL, (2, 1)), _boom)


def _evaluate_batch_wrong_shape(_opened: Session, _inner: WorkerPool) -> None:
    evaluate_batch(_CONFIG, _INITIAL, _sphere)


def _run(_opened: Session, _inner: WorkerPool) -> None:
    run([partial(optimize, _CONFIG, _INITIAL, _boom)] * 2)


@pytest.mark.parametrize(
    "fails",
    [
        pytest.param(_pool_run, id="pool_run"),
        pytest.param(_optimize, id="optimize"),
        pytest.param(_optimize_invalid_config, id="optimize_invalid_config"),
        pytest.param(_evaluate, id="evaluate"),
        pytest.param(_evaluate_invalid_config, id="evaluate_invalid_config"),
        pytest.param(_evaluate_wrong_shape, id="evaluate_wrong_shape"),
        pytest.param(_evaluate_batch, id="evaluate_batch"),
        pytest.param(_evaluate_batch_wrong_shape, id="evaluate_batch_wrong_shape"),
        pytest.param(_run, id="run"),
    ],
)
@pytest.mark.timeout(60)
def test_caught_nested_failure_does_not_abort_the_run(
    fails: Callable[[Session, WorkerPool], None],
) -> None:
    lock = threading.Lock()
    caught: list[type[Exception]] = []

    with session() as opened:
        outer = opened.thread_pool(workers=2)
        inner = opened.thread_pool(workers=2)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            try:
                fails(opened, inner)
            except ValueError as exc:
                with lock:
                    caught.append(type(exc))
            return _sphere(variables, context)

        result = optimize(_CONFIG, _INITIAL, objective, pool=outer)

    assert caught
    assert result.exit_code == ExitCode.FINISHED


def _run_from_an_in_process_objective(
    _opened: Session, work: Callable[[], None]
) -> ExitCode:
    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        work()
        return _sphere(variables, context)

    return optimize(_CONFIG, _INITIAL, objective).exit_code


def _run_from_a_report_callback(opened: Session, work: Callable[[], None]) -> ExitCode:
    def report(_: FunctionResults) -> None:
        work()

    pool = opened.thread_pool(workers=1)
    return optimize(_CONFIG, _INITIAL, _sphere, pool=pool, report=report).exit_code


@pytest.mark.parametrize(
    "optimize",
    [
        pytest.param(_run_from_an_in_process_objective, id="in_process_objective"),
        pytest.param(_run_from_a_report_callback, id="report_callback"),
    ],
)
@pytest.mark.timeout(60)
def test_caught_failure_of_a_pool_run_on_the_optimization_thread_does_not_abort_it(
    optimize: Callable[[Session, Callable[[], None]], ExitCode],
) -> None:
    caught: list[type[Exception]] = []

    with session() as opened:
        inner = opened.thread_pool(workers=1)

        def work() -> None:
            try:
                run(_boom, pool=inner)
            except ValueError as exc:
                caught.append(type(exc))

        exit_code = optimize(opened, work)

    assert caught
    assert exit_code == ExitCode.FINISHED


@pytest.mark.timeout(60)
def test_failing_nested_pool_run_aborts_what_is_nested_in_it() -> None:
    # The middle `run` call has failed before the innermost one starts, so the
    # innermost one is aborted at once instead of running.
    failed = threading.Event()
    done = threading.Event()
    outcomes: list[object] = []

    with session() as opened:
        outer = opened.thread_pool(workers=1)
        middle = opened.thread_pool(workers=2)
        inner = opened.thread_pool(workers=1)

        def _runs_after_the_failure() -> None:
            try:
                assert failed.wait(timeout=30)
                try:
                    outcomes.append(run(partial(int, 1), pool=inner))
                except AbortedError as exc:
                    outcomes.append(exc.exit_code)
            finally:
                done.set()

        work: list[Callable[[], object]] = [_runs_after_the_failure, _boom]

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            with contextlib.suppress(ValueError):
                run(work, pool=middle)
            failed.set()
            return _sphere(variables, context)

        result = evaluate(_CONFIG, _INITIAL, objective, pool=outer)
        assert done.wait(timeout=30)

    assert result.results is not None
    assert outcomes == [ExitCode.ABORTED_ON_ERROR]


def _fails_once_both_evaluate(
    started: threading.Barrier,
) -> Callable[[NDArray[np.float64], EvaluationFunctionContext], float]:
    def objective(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        started.wait()
        msg = "boom"
        raise ValueError(msg)

    return objective


def _records_its_abort(
    started: threading.Barrier, exit_codes: list[ExitCode], recorded: threading.Event
) -> Callable[[NDArray[np.float64], EvaluationFunctionContext], float]:
    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        # Evaluated in-process, so the current scope here is the scope of this
        # optimization.
        scope = current_abort_scope()
        assert scope is not None
        if not recorded.is_set():
            started.wait()
        aborted = threading.Event()
        scope.add_callback(aborted.set)
        assert aborted.wait(timeout=30)
        if not recorded.is_set():
            exit_codes.append(scope.exit_code)
            recorded.set()
        return _sphere(variables, context)

    return objective


@pytest.mark.timeout(60)
def test_failing_optimization_in_a_nested_run_aborts_only_that_call() -> None:
    # Both optimizations are inside their first evaluation when one fails. The
    # `run` call does not wait for the optimization it aborts, so that one
    # records the exit code it was aborted with.
    started = threading.Barrier(2, timeout=30)
    exit_codes: list[ExitCode] = []
    recorded = threading.Event()
    caught: list[type[Exception]] = []

    with session() as opened:
        outer = opened.thread_pool(workers=1)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            optimize_with = partial(optimize, _CONFIG, _INITIAL)
            fails = _fails_once_both_evaluate(started)
            waits = _records_its_abort(started, exit_codes, recorded)
            try:
                run(
                    [partial(optimize_with, fails), partial(optimize_with, waits)],
                    pool=opened.thread_pool(workers=2),
                )
            except ValueError as exc:
                caught.append(type(exc))
            return _sphere(variables, context)

        result = evaluate(_CONFIG, _INITIAL, objective, pool=outer)
        assert recorded.wait(timeout=30)

    assert result.results is not None
    assert caught == [ValueError]
    assert exit_codes == [ExitCode.ABORTED_ON_ERROR]


@pytest.mark.timeout(60)
def test_nested_run_is_aborted_with_its_parent() -> None:
    # The `run` call it is nested in has failed before the call starts, so it is
    # aborted at once instead of running.
    failed = threading.Event()
    done = threading.Event()
    exit_codes: list[ExitCode] = []

    with session() as opened:
        outer = opened.thread_pool(workers=1)
        middle = opened.thread_pool(workers=2)

        def _optimizes_after_the_failure() -> None:
            try:
                assert failed.wait(timeout=30)
                optimization = partial(optimize, _CONFIG, _INITIAL, _sphere)
                run([optimization, optimization])
            except AbortedError as exc:
                exit_codes.append(exc.exit_code)
            finally:
                done.set()

        work: list[Callable[[], object]] = [_optimizes_after_the_failure, _boom]

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            with contextlib.suppress(ValueError):
                run(work, pool=middle)
            failed.set()
            return _sphere(variables, context)

        result = evaluate(_CONFIG, _INITIAL, objective, pool=outer)
        assert done.wait(timeout=30)

    assert result.results is not None
    assert exit_codes == [ExitCode.ABORTED_ON_ERROR]


def _start_plain(target: Callable[[], None]) -> threading.Thread:
    return threading.Thread(target=target)


def _start_in_a_copied_context(target: Callable[[], None]) -> threading.Thread:
    return threading.Thread(target=contextvars.copy_context().run, args=(target,))


@pytest.mark.parametrize(
    "start",
    [
        pytest.param(_start_plain, id="plain"),
        pytest.param(_start_in_a_copied_context, id="copied"),
    ],
)
@pytest.mark.timeout(60)
def test_caught_failure_of_a_pool_run_on_a_thread_of_the_objective_does_not_abort_it(
    start: Callable[[Callable[[], None]], threading.Thread],
) -> None:
    with session() as opened:
        outer = opened.thread_pool(workers=2)
        inner = opened.thread_pool(workers=2)

        def work() -> None:
            with contextlib.suppress(ValueError):
                run(_boom, pool=inner)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            thread = start(work)
            thread.start()
            thread.join(timeout=30)
            return _sphere(variables, context)

        try:
            ended_with = optimize(_CONFIG, _INITIAL, objective, pool=outer).exit_code
        except AbortedError as exc:
            ended_with = exc.exit_code

    assert ended_with == ExitCode.FINISHED
