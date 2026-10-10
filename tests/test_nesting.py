"""Tests for runs and offloads started from inside another run or offload."""

# A run or offload started from code that another one executes is nested in it.
# Its failure is raised to that code, which may handle it; only when the
# exception escapes does the outer one fail and stop the session.

from __future__ import annotations

import contextlib
import contextvars
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt import session
from ropt.components.concurrency import parent_signal
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError, RunsFailedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt import (
        EvaluationFunctionContext,
        OptimizationResult,
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


def _offload(_opened: Session, inner: WorkerPool) -> None:
    inner.offload(_boom)


def _optimize(opened: Session, _inner: WorkerPool) -> None:
    opened.optimize(_CONFIG, _INITIAL, _boom)


def _optimize_invalid_config(opened: Session, _inner: WorkerPool) -> None:
    opened.optimize(_INVALID_CONFIG, _INITIAL, _sphere)


def _evaluate(opened: Session, _inner: WorkerPool) -> None:
    opened.evaluate(_CONFIG, _INITIAL, _boom)


def _evaluate_invalid_config(opened: Session, _inner: WorkerPool) -> None:
    opened.evaluate(_INVALID_CONFIG, _INITIAL, _sphere)


def _evaluate_wrong_shape(opened: Session, _inner: WorkerPool) -> None:
    opened.evaluate(_CONFIG, np.tile(_INITIAL, (2, 1)), _sphere)


def _evaluate_batch(opened: Session, _inner: WorkerPool) -> None:
    opened.evaluate_batch(_CONFIG, np.tile(_INITIAL, (2, 1)), _boom)


def _evaluate_batch_wrong_shape(opened: Session, _inner: WorkerPool) -> None:
    opened.evaluate_batch(_CONFIG, _INITIAL, _sphere)


def _optimize_many(opened: Session, _inner: WorkerPool) -> None:
    opened.optimize_many(_CONFIG, np.tile(_INITIAL, (2, 1)), _boom)


def _optimize_many_mismatched(opened: Session, _inner: WorkerPool) -> None:
    opened.optimize_many(_CONFIG, np.tile(_INITIAL, (2, 1)), [_sphere] * 3)


@pytest.mark.parametrize(
    "fails",
    [
        pytest.param(_offload, id="offload"),
        pytest.param(_optimize, id="optimize"),
        pytest.param(_optimize_invalid_config, id="optimize_invalid_config"),
        pytest.param(_evaluate, id="evaluate"),
        pytest.param(_evaluate_invalid_config, id="evaluate_invalid_config"),
        pytest.param(_evaluate_wrong_shape, id="evaluate_wrong_shape"),
        pytest.param(_evaluate_batch, id="evaluate_batch"),
        pytest.param(_evaluate_batch_wrong_shape, id="evaluate_batch_wrong_shape"),
        pytest.param(_optimize_many, id="optimize_many"),
        pytest.param(_optimize_many_mismatched, id="optimize_many_mismatched"),
    ],
)
@pytest.mark.timeout(60)
def test_caught_nested_failure_does_not_stop_the_run(
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
            except (ValueError, RunsFailedError) as exc:
                with lock:
                    caught.append(type(exc))
            return _sphere(variables, context)

        result = outer.optimize(_CONFIG, _INITIAL, objective)

    assert caught
    assert result.exit_code == ExitCode.FINISHED


def _offload_from_an_in_process_objective(
    opened: Session, offloads: Callable[[], None]
) -> ExitCode:
    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        offloads()
        return _sphere(variables, context)

    return opened.optimize(_CONFIG, _INITIAL, objective).exit_code


def _offload_from_a_report_callback(
    opened: Session, offloads: Callable[[], None]
) -> ExitCode:
    def report(_: FunctionResults) -> None:
        offloads()

    pool = opened.thread_pool(workers=1)
    return pool.optimize(_CONFIG, _INITIAL, _sphere, report=report).exit_code


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(_offload_from_an_in_process_objective, id="in_process_objective"),
        pytest.param(_offload_from_a_report_callback, id="report_callback"),
    ],
)
@pytest.mark.timeout(60)
def test_caught_failure_of_an_offload_on_the_run_thread_does_not_stop_the_run(
    run: Callable[[Session, Callable[[], None]], ExitCode],
) -> None:
    caught: list[type[Exception]] = []

    with session() as opened:
        inner = opened.thread_pool(workers=1)

        def offloads() -> None:
            try:
                inner.offload(_boom)
            except ValueError as exc:
                caught.append(type(exc))

        exit_code = run(opened, offloads)

    assert caught
    assert exit_code == ExitCode.FINISHED


@pytest.mark.timeout(60)
def test_failing_nested_offload_aborts_what_is_nested_in_it() -> None:
    # The middle offload has failed before the innermost one starts, so the
    # innermost one is aborted at once instead of running.
    failed = threading.Event()
    done = threading.Event()
    outcomes: list[object] = []

    with session() as opened:
        outer = opened.thread_pool(workers=1)
        middle = opened.thread_pool(workers=2)
        inner = opened.thread_pool(workers=1)

        def _offloads_after_the_failure() -> None:
            try:
                assert failed.wait(timeout=30)
                try:
                    outcomes.append(inner.offload(partial(int, 1)))
                except AbortedError as exc:
                    outcomes.append(exc.exit_code)
            finally:
                done.set()

        work: list[Callable[[], object]] = [_offloads_after_the_failure, _boom]

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            with contextlib.suppress(ValueError):
                middle.offload(work)
            failed.set()
            return _sphere(variables, context)

        result = outer.evaluate(_CONFIG, _INITIAL, objective)
        assert done.wait(timeout=30)

    assert result.exit_code == ExitCode.FINISHED
    assert outcomes == [ExitCode.ABORTED_ON_ERROR]


def _waits_until_aborted(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    # Evaluated in-process, so the parent signal here is the signal of this run.
    signal = parent_signal()
    assert signal is not None
    aborted = threading.Event()
    signal.add_callback(aborted.set)
    assert aborted.wait(timeout=30)
    return _sphere(variables, context)


@pytest.mark.timeout(60)
def test_failing_run_of_a_nested_optimize_many_aborts_only_the_runs_of_that_call() -> (
    None
):
    outcomes: list[tuple[OptimizationResult | Exception, ...]] = []

    with session() as opened:
        outer = opened.thread_pool(workers=1)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            try:
                opened.optimize_many(
                    _CONFIG, np.tile(_INITIAL, (2, 1)), [_boom, _waits_until_aborted]
                )
            except RunsFailedError as exc:
                outcomes.append(exc.outcomes)
            return _sphere(variables, context)

        result = outer.optimize(_CONFIG, _INITIAL, objective)

    assert result.exit_code == ExitCode.FINISHED
    assert outcomes
    for failed, aborted in outcomes:
        assert isinstance(failed, ValueError)
        assert not isinstance(aborted, Exception)
        assert aborted.exit_code == ExitCode.ABORTED_ON_ERROR


@pytest.mark.timeout(60)
def test_nested_optimize_many_is_aborted_with_its_parent() -> None:
    # The offload it is nested in has failed before the call starts, so its runs
    # are aborted at once instead of running.
    failed = threading.Event()
    done = threading.Event()
    exit_codes: list[ExitCode] = []

    with session() as opened:
        outer = opened.thread_pool(workers=1)
        middle = opened.thread_pool(workers=2)

        def _optimizes_after_the_failure() -> None:
            try:
                assert failed.wait(timeout=30)
                results = opened.optimize_many(
                    _CONFIG, np.tile(_INITIAL, (2, 1)), _sphere
                )
                exit_codes.extend(result.exit_code for result in results)
            finally:
                done.set()

        work: list[Callable[[], object]] = [_optimizes_after_the_failure, _boom]

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            with contextlib.suppress(ValueError):
                middle.offload(work)
            failed.set()
            return _sphere(variables, context)

        result = outer.evaluate(_CONFIG, _INITIAL, objective)
        assert done.wait(timeout=30)

    assert result.exit_code == ExitCode.FINISHED
    assert exit_codes == [ExitCode.ABORTED_ON_ERROR] * 2


def _start_plain(target: Callable[[], None]) -> threading.Thread:
    return threading.Thread(target=target)


def _start_in_a_copied_context(target: Callable[[], None]) -> threading.Thread:
    return threading.Thread(target=contextvars.copy_context().run, args=(target,))


@pytest.mark.parametrize(
    ("start", "exit_code"),
    [
        pytest.param(_start_plain, ExitCode.ABORTED_ON_ERROR, id="plain"),
        pytest.param(_start_in_a_copied_context, ExitCode.FINISHED, id="copied"),
    ],
)
@pytest.mark.timeout(60)
def test_offload_from_a_thread_of_the_objective_is_nested_only_in_its_context(
    start: Callable[[Callable[[], None]], threading.Thread], exit_code: ExitCode
) -> None:
    with session() as opened:
        outer = opened.thread_pool(workers=2)
        inner = opened.thread_pool(workers=2)

        def offloads() -> None:
            with contextlib.suppress(ValueError):
                inner.offload(_boom)

        def objective(
            variables: NDArray[np.float64], context: EvaluationFunctionContext
        ) -> float:
            thread = start(offloads)
            thread.start()
            thread.join(timeout=30)
            return _sphere(variables, context)

        result = outer.optimize(_CONFIG, _INITIAL, objective)

    assert result.exit_code == exit_code
