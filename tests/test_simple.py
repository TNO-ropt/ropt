"""Tests for the sequential high-level ``optimize`` API."""

# Two things here are easy to "simplify" and must not be; the other traps in
# this file are explained where they sit.
#
# - test_execution_block_refuses_reentry has to match the message. Asserting
#   `WorkflowError` alone passes with the guard removed, because
#   open_executor's "Only one execution block" surfaces instead.
# - The monkeypatched tests name their target as an attribute and assert it was
#   used. An earlier version patched a method by string and became a no-op the
#   day it was renamed, which mypy cannot see.

from __future__ import annotations

import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.compute_steps import EvaluationStep
from ropt.components.evaluators import FunctionEvaluator
from ropt.components.event_handlers import EventDispatcher
from ropt.components.executors import ThreadingExecutor
from ropt.enums import ExitCode
from ropt.exceptions import ExecutorStopped, WorkflowError
from ropt.simple import (
    EvaluateResult,
    EvaluationFunctionContext,
    HistoryHandler,
    OptimizeResult,
    _blocks,
    can_offload,
    compose,
    evaluate,
    evaluate_many,
    handlers,
    hpc,
    offload,
    optimize,
    optimize_many,
    processes,
    threads,
)
from ropt.simple._function import adapt_function
from ropt.simple._handlers import _handler_stack, current_handlers
from ropt.simple._session import _Session, current_executor, current_session

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

try:
    import cloudpickle  # ruff: ignore[unused-import]
    import pandas as pd
    import pysqa  # ruff: ignore[unused-import]

    from ropt.components.executors.__main__ import run_task

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False

initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {"max_functions": 20},
        "backend": {
            "method": "slsqp",
            "max_iterations": 15,
            "convergence_tolerance": 1e-5,
        },
        "variables": {
            "variable_count": initial_values.size,
            "perturbation_magnitudes": 0.01,
        },
    }


def test_optimize_returns_run_result(config: Any, test_functions: Any) -> None:
    result = optimize(config, initial_values, test_functions[0])
    assert isinstance(result, OptimizeResult)
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED
    assert result.variables is not None
    assert np.allclose(result.variables, 0.5, atol=0.02)
    assert result.target_objective is not None
    assert result.target_objective == pytest.approx(0.0, abs=1e-3)
    assert result.objectives is not None
    assert result.objectives.shape == (1,)
    assert result.constraints is None
    assert result.results is not None


def test_optimize_accepts_evaluation_function_result(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    result = optimize(config, initial_values, eval_func([test_functions[0]]))
    assert result.variables is not None
    assert np.allclose(result.variables, 0.5, atol=0.02)


def test_optimize_accepts_sequence_for_multiple_objectives(
    config: Any, test_functions: Any
) -> None:
    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> list[float]:
        return [func(variables, context) for func in test_functions]

    config["objectives"] = {"weights": [0.75, 0.25]}
    result = optimize(config, initial_values, objective)
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optimize_no_valid_result_has_none_fields(config: Any) -> None:
    result = optimize(config, initial_values, lambda _v, _c: np.nan)
    assert result.exit_code == ExitCode.TOO_FEW_REALIZATIONS
    assert result.variables is None
    assert result.target_objective is None
    assert result.objectives is None
    assert result.constraints is None
    assert result.results is None


def test_optimize_attaches_metadata_to_results(
    config: Any, test_functions: Any
) -> None:
    result = optimize(
        config, initial_values, test_functions[0], metadata={"tag": "run-a"}
    )
    assert result.results is not None
    assert result.results.metadata["tag"] == "run-a"


def test_optimize_local_handler_collects_results(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[history])
    assert history["results"] is not None
    assert len(history["results"]) > 0


def test_optimize_report_callback_receives_evaluate_results(
    config: Any, test_functions: Any
) -> None:
    reported: list[EvaluateResult] = []
    optimize(config, initial_values, test_functions[0], report=reported.append)
    assert reported
    assert all(isinstance(item, EvaluateResult) for item in reported)
    assert any(item.target_objective is not None for item in reported)


def test_handlers_report_callback_reports_across_the_block(
    config: Any, test_functions: Any
) -> None:
    reported: list[EvaluateResult] = []
    with handlers(report=reported.append):
        optimize(config, initial_values, test_functions[0])
        after_first = len(reported)
        optimize(config, initial_values, test_functions[0])
    assert after_first > 0
    assert len(reported) > after_first
    assert all(isinstance(item, EvaluateResult) for item in reported)


def test_threaded_handlers_run_in_thread() -> None:
    loop_handler = HistoryHandler()
    io_handler = HistoryHandler()
    scope = handlers(loop_handler, threaded=io_handler)
    with scope:
        in_thread = dict(scope._running_dispatcher._handlers)  # ruff: ignore[private-member-access]
    assert in_thread[loop_handler] is False
    assert in_thread[io_handler] is True


def test_threaded_accepts_handler_sequence() -> None:
    first = HistoryHandler()
    second = HistoryHandler()
    scope = handlers(threaded=[first, second])
    with scope:
        in_thread = dict(scope._running_dispatcher._handlers)  # ruff: ignore[private-member-access]
    assert in_thread[first] is True
    assert in_thread[second] is True


def test_threaded_handler_keeps_flag_when_inherited() -> None:
    io_handler = HistoryHandler()
    outer = handlers(threaded=io_handler)
    with outer:
        inner = handlers()
        with inner:
            assert dict(inner._running_dispatcher._handlers)[io_handler] is True  # ruff: ignore[private-member-access]
        assert dict(outer._running_dispatcher._handlers)[io_handler] is True  # ruff: ignore[private-member-access]


def test_relisting_handler_overrides_threaded_flag() -> None:
    handler = HistoryHandler()
    outer = handlers(threaded=handler)
    with outer:
        inner = handlers(handler)  # re-listed as a loop-thread handler
        with inner:
            assert dict(inner._running_dispatcher._handlers)[handler] is False  # ruff: ignore[private-member-access]
        assert dict(outer._running_dispatcher._handlers)[handler] is True  # ruff: ignore[private-member-access]


def test_handlers_block_binds_scope() -> None:
    handler = HistoryHandler()
    with handlers(handler) as scope:
        assert scope is current_handlers()


def test_empty_handlers_block_adds_no_forwarding_handler(
    config: Any, test_functions: Any
) -> None:
    with handlers() as scope:
        step = EvaluationStep(
            evaluator=FunctionEvaluator(
                function=adapt_function(test_functions[0], 1, 0)
            )
        )
        # Nothing wants events, so the run must not be given a forwarding handler.
        scope.attach_to(step)
        assert step.event_handlers == []
        result = optimize(config, initial_values, test_functions[0])
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


def test_handler_reusable_after_scope_closes() -> None:
    handler = HistoryHandler()
    with handlers(handler):
        pass
    scope = handlers(handler)
    with scope:
        assert set(dict(scope._running_dispatcher._handlers)) == {handler}  # ruff: ignore[private-member-access]


def test_handler_reusable_after_scope_fails_to_open() -> None:
    good = HistoryHandler()
    bad = HistoryHandler()
    with (
        pytest.raises(WorkflowError, match="already registered with a dispatcher"),
        handlers(good, bad, bad),
    ):
        pass
    scope = handlers(good, bad)
    with scope:
        assert set(dict(scope._running_dispatcher._handlers)) == {good, bad}  # ruff: ignore[private-member-access]


def test_handlers_block_refuses_reentry(config: Any, test_functions: Any) -> None:
    handler = HistoryHandler()
    scope = handlers(handler)
    with pytest.raises(WorkflowError, match="already open"), scope, scope:
        pass
    # The rejected re-entry must leave nothing behind: a scope stranded on the
    # stack would feed every later run to a dispatcher that no longer runs.
    assert _handler_stack.get() == ()
    with handlers(handler):
        optimize(config, initial_values, test_functions[0])
    assert len(handler["results"]) > 0


def test_execution_block_refuses_reentry(config: Any, test_functions: Any) -> None:
    scope = threads(workers=1)
    with pytest.raises(WorkflowError, match="already open"), scope, scope:
        pass
    # The rejected re-entry must leave the first block's state alone, so the
    # scope still releases its session and can be opened again.
    assert current_session() is None
    with scope:
        result = optimize(config, initial_values, test_functions[0])
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


def test_execution_block_survives_a_failed_session_acquire(
    config: Any, test_functions: Any, monkeypatch: Any
) -> None:
    # A scope that fails before it acquires anything has nothing to release, and
    # must not refuse every later block on the strength of an entry that opened
    # nothing. Reachable: _acquire_session starts a loop and a thread.
    calls = 0

    def _boom() -> tuple[_Session, None]:
        nonlocal calls
        calls += 1
        msg = "no thread for you"
        raise RuntimeError(msg)

    scope = threads(workers=1)
    monkeypatch.setattr(_blocks, "_acquire_session", _boom)
    with pytest.raises(RuntimeError, match="no thread for you"), scope:
        pass
    assert calls == 1
    monkeypatch.undo()
    with scope:
        result = optimize(config, initial_values, test_functions[0])
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


def test_closed_execution_block_releases_session() -> None:
    # A scope that keeps a stopped session keeps its loop and its thread with
    # it, and would hand a dead one to a second use of the same object.
    scope = threads(workers=1)
    with scope:
        assert scope._session is not None  # ruff: ignore[private-member-access]
    assert scope._session is None  # ruff: ignore[private-member-access]
    assert scope._token is None  # ruff: ignore[private-member-access]
    with scope:
        assert scope._session is not None  # ruff: ignore[private-member-access]


def test_local_handler_refused_by_block(config: Any, test_functions: Any) -> None:
    handler = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[handler])
    with (
        pytest.raises(WorkflowError, match="already in use") as excinfo,
        handlers(handler),
    ):
        pass
    # The low-level refusal names a compute step, which this API never hands
    # out; a reader cannot act on it.
    assert "compute step" not in str(excinfo.value)
    assert "separate handler" in str(excinfo.value)


def test_closed_handlers_block_reopens() -> None:
    handler = HistoryHandler()
    scope = handlers(handler)
    with scope:
        assert set(dict(scope._running_dispatcher._handlers)) == {handler}  # ruff: ignore[private-member-access]
    assert scope._current_handlers == set()  # ruff: ignore[private-member-access]
    with scope:
        assert set(dict(scope._running_dispatcher._handlers)) == {handler}  # ruff: ignore[private-member-access]
    assert scope._current_handlers == set()  # ruff: ignore[private-member-access]


def test_handlers_block_releases_session_on_rollback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _failing_remove(*_args: object, **_kwargs: object) -> None:
        msg = "rollback failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(EventDispatcher, "remove_event_handler", _failing_remove)
    handler = HistoryHandler()
    with (
        pytest.raises(RuntimeError, match="rollback failed"),
        handlers(handler, handler),
    ):
        pass
    assert current_session() is None


def test_failing_block_teardown_releases_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _failing_close(_self: object) -> None:
        msg = "teardown failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(_Session, "close_executor", _failing_close)
    with pytest.raises(RuntimeError, match="teardown failed"), threads(workers=1):
        pass
    assert current_session() is None


def test_optimize_local_handler_accumulates_across_sequential_calls(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[history])
    after_first = len(history["results"])
    optimize(config, initial_values, test_functions[0], handlers=[history])
    assert len(history["results"]) > after_first


def test_optimize_local_handler_released_after_run(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[history])
    assert history._claimed is False  # ruff: ignore[private-member-access]


def test_optimize_local_handler_released_after_error(config: Any) -> None:
    history = HistoryHandler()

    def _boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        optimize(config, initial_values, _boom, handlers=[history])
    assert history._claimed is False  # ruff: ignore[private-member-access]


def test_optimize_local_handler_claimed_during_run(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    observed: list[bool] = []

    def _report(_: EvaluateResult) -> None:
        observed.append(history._claimed)  # ruff: ignore[private-member-access]
        with pytest.raises(WorkflowError, match="already been claimed"):
            history.claim()

    optimize(
        config,
        initial_values,
        test_functions[0],
        handlers=[history],
        report=_report,
    )
    assert observed
    assert all(observed)
    assert history._claimed is False  # ruff: ignore[private-member-access]


def test_optimize_local_handler_claim_rolls_back_on_failure(
    config: Any, test_functions: Any
) -> None:
    first = HistoryHandler()
    second = HistoryHandler()
    second.claim()  # stands in for a handler already in use by another run
    with pytest.raises(WorkflowError, match="already been claimed"):
        optimize(config, initial_values, test_functions[0], handlers=[first, second])
    assert first._claimed is False  # ruff: ignore[private-member-access]


def test_report_callback_stops_optimization(config: Any, test_functions: Any) -> None:
    reported = 0

    def _report(_: EvaluateResult) -> bool:
        nonlocal reported
        reported += 1
        return True

    result = optimize(config, initial_values, test_functions[0], report=_report)
    assert result.exit_code == ExitCode.USER_ABORT
    assert reported == 1


def test_report_callback_stops_only_own_run(config: Any, test_functions: Any) -> None:
    def _stop(_: EvaluateResult) -> bool:
        return True

    def _continue(_: EvaluateResult) -> None:
        return None

    x0 = np.array([initial_values, initial_values])
    with threads(workers=2):
        results = optimize_many(
            config, x0, test_functions[0], report=[_stop, _continue]
        )
    assert results[0].exit_code == ExitCode.USER_ABORT
    assert results[1].exit_code != ExitCode.USER_ABORT


def test_adapt_function_rejects_scalar_for_multiple_objectives() -> None:
    callback = adapt_function(lambda _v, _c: 1.0, n_obj=2, n_con=0)
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, batch_id=0, eval_idx=0
    )
    with pytest.raises(ValueError, match="scalar return value"):
        callback(np.zeros(2), context)


def test_adapt_function_rejects_wrong_shape() -> None:
    callback = adapt_function(lambda _v, _c: [1.0, 2.0, 3.0], n_obj=1, n_con=1)
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, batch_id=0, eval_idx=0
    )
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        callback(np.zeros(2), context)


def test_adapt_function_splits_objectives_and_constraints() -> None:
    callback = adapt_function(lambda _v, _c: [1.0, 2.0, 3.0], n_obj=1, n_con=2)
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, batch_id=0, eval_idx=0
    )
    result = callback(np.zeros(2), context)
    assert np.array_equal(result.objectives, [1.0])
    assert result.constraints is not None
    assert np.array_equal(result.constraints, [2.0, 3.0])


def test_evaluate_single_vector(config: Any, test_functions: Any) -> None:
    result = evaluate(config, initial_values, test_functions[0])
    assert isinstance(result, EvaluateResult)
    assert result.target_objective is not None
    assert result.target_objective == pytest.approx(0.66)
    assert result.objectives is not None
    assert result.objectives.shape == (1,)
    assert result.constraints is None
    assert result.results.evaluations.variables.shape == (initial_values.size,)


def test_user_thread_does_not_inherit_open_blocks(
    config: Any, test_functions: Any
) -> None:
    handler = HistoryHandler()
    seen: dict[str, Any] = {}

    def worker() -> None:
        seen["session"] = current_session()
        seen["executor"] = current_executor()
        seen["handlers"] = current_handlers()
        optimize(config, initial_values, test_functions[0])

    with threads(workers=2), handlers(handler):
        assert current_session() is not None
        assert current_executor() is not None
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen["session"] is None
    assert seen["executor"] is None
    assert seen["handlers"] is None
    assert handler["results"] is None


def test_evaluate_feeds_a_shared_handlers_block(
    config: Any, test_functions: Any
) -> None:
    handler = HistoryHandler()
    with handlers(handler):
        evaluate(config, initial_values, test_functions[0])
        evaluate(config, np.zeros(initial_values.size), test_functions[0])
    assert len(handler["results"]) == 2


def test_evaluate_rejects_matrix(config: Any, test_functions: Any) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    with pytest.raises(ValueError, match="single vector"):
        evaluate(config, matrix, test_functions[0])


def test_evaluate_many_returns_result_per_row(config: Any, test_functions: Any) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    results = evaluate_many(config, matrix, test_functions[0])
    assert len(results) == 2
    assert all(isinstance(result, EvaluateResult) for result in results)
    # Squared distance to [0.5, 0.5, 0.5]: row 0 = 0.5^2+0.5^2+0.4^2, row 1 = 3*0.5^2.
    for result, expected in zip(results, [0.66, 0.75], strict=True):
        assert result.target_objective == pytest.approx(expected)


def test_evaluate_many_single_row(config: Any, test_functions: Any) -> None:
    results = evaluate_many(config, initial_values.reshape(1, -1), test_functions[0])
    assert len(results) == 1
    assert results[0].target_objective == pytest.approx(0.66)


def test_evaluate_many_rejects_vector(config: Any, test_functions: Any) -> None:
    with pytest.raises(ValueError, match="2-D matrix"):
        evaluate_many(config, initial_values, test_functions[0])


def test_evaluate_multiple_objectives(config: Any, eval_func: Any) -> None:
    config["objectives"] = {"weights": [0.75, 0.25]}
    result = evaluate(config, initial_values, eval_func())
    assert result.objectives is not None
    assert result.objectives.shape == (2,)
    assert result.constraints is None


def test_evaluate_attaches_metadata_to_results(
    config: Any, test_functions: Any
) -> None:
    result = evaluate(
        config, initial_values, test_functions[0], metadata={"tag": "eval"}
    )
    assert result.results is not None
    assert result.results.metadata["tag"] == "eval"


def test_evaluate_many_attaches_metadata_to_every_result(
    config: Any, test_functions: Any
) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    results = evaluate_many(config, matrix, test_functions[0], metadata={"tag": "eval"})
    assert len(results) == 2
    for result in results:
        assert result.results is not None
        assert result.results.metadata["tag"] == "eval"


def test_optimize_with_threads(config: Any, test_functions: Any) -> None:
    with threads(workers=2):
        result = optimize(config, initial_values, test_functions[0])
    assert result.variables is not None
    assert np.allclose(result.variables, 0.5, atol=0.02)


def _collect_batch_ids(sink: list[int], lock: threading.Lock) -> Any:
    def _function(
        _variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        with lock:
            sink.append(context.batch_id)
        return 0.0

    return _function


def test_batch_ids_are_unique_across_runs_in_an_execution_block(config: Any) -> None:
    first: list[int] = []
    second: list[int] = []
    lock = threading.Lock()
    with threads(workers=2):
        optimize(config, initial_values, _collect_batch_ids(first, lock))
        optimize(config, initial_values, _collect_batch_ids(second, lock))
    assert first
    assert second
    assert not set(first) & set(second)


def test_batch_ids_are_unique_across_concurrent_runs(config: Any) -> None:
    sinks: list[list[int]] = [[], [], []]
    lock = threading.Lock()
    with threads(workers=2):
        optimize_many(
            config,
            initial_values,
            [_collect_batch_ids(sink, lock) for sink in sinks],
        )
    for sink in sinks:
        assert sink
    assert sum(len(set(sink)) for sink in sinks) == len(set().union(*sinks))


def test_batch_ids_restart_for_each_execution_block(config: Any) -> None:
    first: list[int] = []
    second: list[int] = []
    lock = threading.Lock()
    with threads(workers=2):
        optimize(config, initial_values, _collect_batch_ids(first, lock))
    with threads(workers=2):
        optimize(config, initial_values, _collect_batch_ids(second, lock))
    assert min(first) == 0
    assert min(second) == 0


def test_batch_ids_restart_for_each_run_without_an_execution_block(
    config: Any,
) -> None:
    first: list[int] = []
    second: list[int] = []
    lock = threading.Lock()
    optimize(config, initial_values, _collect_batch_ids(first, lock))
    optimize(config, initial_values, _collect_batch_ids(second, lock))
    assert min(first) == 0
    assert min(second) == 0


def _record_metadata(sink: list[Any], lock: threading.Lock) -> Any:
    def _function(
        _variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        with lock:
            sink.append(context.metadata)
        return 0.0

    return _function


def test_metadata_reaches_the_evaluation_function(config: Any) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    optimize(config, initial_values, _record_metadata(seen, lock), metadata={"run": 7})
    assert seen
    assert all(item == {"run": 7} for item in seen)


def test_metadata_is_none_in_the_evaluation_function_when_not_given(
    config: Any,
) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    optimize(config, initial_values, _record_metadata(seen, lock))
    assert seen
    assert all(item is None for item in seen)


def test_metadata_reaches_the_evaluation_function_with_threads(config: Any) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    with threads(workers=2):
        optimize(
            config, initial_values, _record_metadata(seen, lock), metadata={"run": 7}
        )
    assert seen
    assert all(item == {"run": 7} for item in seen)


def test_metadata_reaches_the_evaluation_function_of_evaluate(config: Any) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    evaluate(config, initial_values, _record_metadata(seen, lock), metadata={"run": 7})
    assert seen
    assert all(item == {"run": 7} for item in seen)


def test_metadata_per_run_reaches_each_evaluation_function(config: Any) -> None:
    first: list[Any] = []
    second: list[Any] = []
    lock = threading.Lock()
    with threads(workers=2):
        optimize_many(
            config,
            initial_values,
            [_record_metadata(first, lock), _record_metadata(second, lock)],
            metadata=[{"run": 0}, {"run": 1}],
        )
    assert all(item == {"run": 0} for item in first)
    assert all(item == {"run": 1} for item in second)


def test_evaluate_many_with_threads(config: Any, test_functions: Any) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2):
        results = evaluate_many(config, matrix, test_functions[0])
    for result, expected in zip(results, [0.66, 0.75], strict=True):
        assert result.target_objective == pytest.approx(expected)


def test_nested_execution_blocks_raise() -> None:
    with (
        threads(workers=1),
        pytest.raises(WorkflowError, match="Only one execution block"),
        threads(workers=1),
    ):
        pass


_INNER_CONFIG: dict[str, Any] = {
    "optimizer": {"max_functions": 3},
    "backend": {"method": "slsqp", "max_iterations": 2},
    "variables": {
        "variable_count": initial_values.size,
        "perturbation_magnitudes": 0.01,
    },
}


def _sphere(variables: NDArray[np.float64], _context: Any) -> float:
    return float(variables @ variables)


def _run_inner_optimization(variables: NDArray[np.float64], _context: Any) -> float:
    with threads(workers=1):
        result = optimize(_INNER_CONFIG, variables, _sphere)
    assert result.target_objective is not None
    return result.target_objective


def _offload_from_evaluation(_variables: NDArray[np.float64], _context: Any) -> float:
    return float(offload(_return_one))


def _return_one() -> float:
    return 1.0


def test_isolated_nested_block_in_threads_evaluation(config: Any) -> None:
    with threads(workers=2):
        result = optimize(config, initial_values, _run_inner_optimization)
    assert result.variables is not None
    assert result.target_objective is not None


@pytest.mark.slow
def test_isolated_nested_block_in_processes_evaluation(config: Any) -> None:
    with processes(workers=2):
        result = optimize(config, initial_values, _run_inner_optimization)
    assert result.variables is not None
    assert result.target_objective is not None


_BILEVEL_CONFIG: dict[str, Any] = {
    "optimizer": {"max_functions": 20},
    "backend": {"method": "slsqp", "max_iterations": 15, "convergence_tolerance": 1e-6},
    "variables": {"variable_count": 1, "perturbation_magnitudes": 0.01},
}


def _inner_objective(
    variables: NDArray[np.float64], _context: Any, outer_value: float
) -> float:
    b = float(variables[0])
    return (outer_value - 2.0) ** 2 + (b - 3.0) ** 2


def _bilevel_outer(variables: NDArray[np.float64], _context: Any) -> float:
    a = float(variables[0])
    with threads(workers=1):
        inner = optimize(
            _BILEVEL_CONFIG, [0.0], partial(_inner_objective, outer_value=a)
        )
    assert inner.target_objective is not None
    return inner.target_objective


def test_isolated_nested_optimization_on_threads() -> None:
    with threads(workers=1):
        result = optimize(_BILEVEL_CONFIG, [0.0], _bilevel_outer)
    assert result.variables is not None
    assert result.variables[0] == pytest.approx(2.0, abs=0.05)
    assert result.target_objective == pytest.approx(0.0, abs=1e-2)


def test_offload_in_evaluation_finds_no_executor(config: Any) -> None:
    with (
        threads(workers=2),
        pytest.raises(WorkflowError, match="found no executor"),
    ):
        optimize(config, initial_values, _offload_from_evaluation)


def test_gather_shared_activates_the_session_on_driver_threads() -> None:
    def _square(value: int) -> int:
        return value * value

    with threads(workers=2):
        functions = [partial(_square, i) for i in (1, 2, 3)]
        [result] = compose.gather_shared([lambda: offload(functions)], limit=1)
    assert result == (1, 4, 9)


def test_gather_shared_requires_execution_block() -> None:
    with pytest.raises(WorkflowError, match="requires an execution block"):
        compose.gather_shared([lambda: 1])


def test_gather_shared_does_not_propagate_handler_scope() -> None:
    seen: list[bool] = []

    def _look() -> bool:
        seen.append(compose.current_handlers() is not None)
        return True

    with threads(workers=1), handlers(report=lambda _: None):
        assert compose.current_handlers() is not None
        assert current_executor() is not None
        compose.gather_shared([_look], limit=1)
    assert seen == [False]


class _FatalWork(BaseException):
    """Not an Exception, so the worker loops let it reach the task group."""


def _fatal_work() -> int:
    msg = "worker died"
    raise _FatalWork(msg)


def test_fatal_worker_error_reported_on_block_exit() -> None:
    def _run_block() -> None:
        with threads(workers=1), pytest.raises(ExecutorStopped):
            offload(_fatal_work)

    with pytest.raises(BaseExceptionGroup) as excinfo:
        _run_block()
    matched, _ = excinfo.value.split(_FatalWork)
    assert matched is not None


def _double(value: float) -> float:
    return 2.0 * value


def _offload_in_own_block(variables: NDArray[np.float64], _context: Any) -> float:
    with threads(workers=2):
        doubled = offload([partial(_double, 3.0), partial(_double, 4.0)])
    assert doubled == (6.0, 8.0)
    return float(variables @ variables)


def _assert_cannot_offload(variables: NDArray[np.float64], _context: Any) -> float:
    assert can_offload() is False
    return float(variables @ variables)


def _own_handlers_in_evaluation(variables: NDArray[np.float64], _context: Any) -> float:
    history = HistoryHandler()
    with handlers(history):
        optimize(_INNER_CONFIG, variables, _sphere)
    assert len(history.results) > 0
    return float(variables @ variables)


def test_offload_in_evaluation_uses_its_own_block(config: Any) -> None:
    with threads(workers=2):
        result = optimize(config, initial_values, _offload_in_own_block)
    assert result.target_objective is not None


def test_can_offload_is_false_in_an_evaluation(config: Any) -> None:
    with threads(workers=2):
        result = optimize(config, initial_values, _assert_cannot_offload)
    assert result.target_objective is not None


def test_handlers_block_in_an_evaluation(config: Any) -> None:
    with threads(workers=2):
        result = optimize(config, initial_values, _own_handlers_in_evaluation)
    assert result.target_objective is not None


@pytest.mark.slow
def test_sequential_execution_managers_are_allowed(
    config: Any, test_functions: Any
) -> None:
    with threads(workers=2):
        first = optimize(config, initial_values, test_functions[0])
    with processes(workers=2):
        second = optimize(config, initial_values, test_functions[0])
    assert first.exit_code == second.exit_code
    assert first.variables == pytest.approx(second.variables)


def test_session_without_task_group_reports_stopped() -> None:
    session = _Session()
    with pytest.raises(WorkflowError, match="is not running"):
        session.open_dispatcher(EventDispatcher())
    with pytest.raises(WorkflowError, match="is not running"):
        session.open_executor(lambda: ThreadingExecutor(workers=1))


def test_session_clears_after_block(config: Any, test_functions: Any) -> None:
    with threads(workers=1):
        pass
    result = optimize(config, initial_values, test_functions[0])
    assert result.variables is not None
    assert np.allclose(result.variables, 0.5, atol=0.02)


def test_threads_objective_exception_propagates(config: Any) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"), threads(workers=2):
        optimize(config, initial_values, boom)


def test_threads_session_survives_objective_exception(
    config: Any, test_functions: Any
) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    with threads(workers=2):
        with pytest.raises(ValueError, match="boom"):
            optimize(config, initial_values, boom)
        # The session survives; the pool is re-established on the next run.
        result = optimize(config, initial_values, test_functions[0])
        assert result.variables is not None
        assert np.allclose(result.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_optimize_with_processes(config: Any, test_functions: Any) -> None:
    with processes(workers=2):
        result = optimize(config, initial_values, test_functions[0])
    assert result.variables is not None
    assert np.allclose(result.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_processes_without_cloudpickle(config: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor._HAVE_CLOUDPICKLE",
        False,
    )
    with processes(workers=2):
        result = optimize(config, initial_values, _sphere)
    assert result.variables is not None
    assert np.allclose(result.variables, 0.0, atol=0.02)


@pytest.mark.slow
def test_evaluate_without_cloudpickle(config: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor._HAVE_CLOUDPICKLE",
        False,
    )
    with processes(workers=2):
        result = evaluate(config, initial_values, _sphere)
    assert result.target_objective == pytest.approx(0.01)


def test_optimize_many_broadcasts_config_and_objective(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2):
        results = optimize_many(config, starts, test_functions[0])
    assert len(results) == 2
    assert all(isinstance(result, OptimizeResult) for result in results)
    for result in results:
        assert result.variables is not None
        assert np.allclose(result.variables, 0.5, atol=0.02)


def test_optimize_many_per_run_objectives(config: Any, test_functions: Any) -> None:
    with threads(workers=2):
        results = optimize_many(
            config, initial_values, [test_functions[0], test_functions[1]]
        )
    assert len(results) == 2
    assert results[0].variables is not None
    assert results[1].variables is not None
    assert np.allclose(results[0].variables, [0.5, 0.5, 0.5], atol=0.02)
    assert np.allclose(results[1].variables, [-1.5, -1.5, 0.5], atol=0.02)


def test_optimize_many_report_callback_shared_across_runs(
    config: Any, test_functions: Any
) -> None:
    reported: list[EvaluateResult] = []
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2):
        optimize_many(config, starts, test_functions[0], report=reported.append)
    assert reported
    assert all(isinstance(item, EvaluateResult) for item in reported)


def test_optimize_many_accepts_a_report_per_run(
    config: Any, test_functions: Any
) -> None:
    first: list[EvaluateResult] = []
    second: list[EvaluateResult] = []
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2):
        optimize_many(
            config, starts, test_functions[0], report=[first.append, second.append]
        )
    assert first
    assert second
    assert all(isinstance(item, EvaluateResult) for item in (*first, *second))


def test_optimize_many_rejects_mismatched_report_sequence(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2), pytest.raises(ValueError, match="number of runs"):
        optimize_many(config, starts, test_functions[0], report=[lambda _r: None])


def test_optimize_many_broadcasts_a_single_metadata_dict(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2):
        results = optimize_many(
            config, starts, test_functions[0], metadata={"group": "g"}
        )
    for result in results:
        assert result.results is not None
        assert result.results.metadata["group"] == "g"


def test_optimize_many_accepts_metadata_per_run(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2):
        results = optimize_many(
            config,
            starts,
            test_functions[0],
            metadata=[{"run_id": 0}, {"run_id": 1}],
        )
    assert len(results) == 2
    for idx, result in enumerate(results):
        assert result.results is not None
        assert result.results.metadata["run_id"] == idx


def test_optimize_many_rejects_mismatched_metadata_sequence(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2), pytest.raises(ValueError, match="number of runs"):
        optimize_many(config, starts, test_functions[0], metadata=[{"run_id": 0}])


def test_optimize_many_respects_limit(config: Any, test_functions: Any) -> None:
    starts = np.tile(initial_values, (4, 1))
    with threads(workers=2):
        results = optimize_many(config, starts, test_functions[0], limit=2)
    assert len(results) == 4


def test_optimize_many_mismatched_lengths_raises(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2), pytest.raises(ValueError, match="same length"):
        optimize_many(
            config, starts, [test_functions[0], test_functions[1], test_functions[0]]
        )


def test_optimize_many_rejects_more_than_two_dimensional_starts(
    config: Any, test_functions: Any
) -> None:
    starts = np.tile(initial_values, (2, 2, 1))
    with threads(workers=2), pytest.raises(ValueError, match="vector or a 2-D matrix"):
        optimize_many(config, starts, test_functions[0])


def test_optimize_many_without_runs_returns_nothing(test_functions: Any) -> None:
    with threads(workers=2):
        assert optimize_many([], initial_values, test_functions[0]) == ()


def test_optimize_many_requires_session(config: Any, test_functions: Any) -> None:
    with pytest.raises(WorkflowError, match="requires an execution block"):
        optimize_many(config, initial_values, test_functions[0])


def test_optimize_many_fail_fast(config: Any, test_functions: Any) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    starts = np.tile(initial_values, (3, 1))
    with threads(workers=2), pytest.raises(ValueError, match="boom"):
        optimize_many(config, starts, [test_functions[0], boom, test_functions[0]])


@pytest.mark.timeout(60)
def test_optimize_many_skips_runs_that_have_not_started(config: Any) -> None:
    calls = 0
    lock = threading.Lock()
    ran_again = threading.Event()

    def boom(_v: Any, _c: Any) -> float:
        nonlocal calls
        with lock:
            calls += 1
            if calls > 1:
                ran_again.set()
        msg = "boom"
        raise ValueError(msg)

    # One at a time, so the first run to be let through fails before any other
    # is admitted, and `run_concurrent` sets its stop flag inside the slot
    # before releasing it. A pending run therefore cannot start -- but that is
    # a negative, and the runs that would disprove it are released just after
    # the failure propagates, so asserting straight away proves nothing. Wait
    # on the event a second run would set: it returns the moment one does, and
    # only costs the ceiling when no run does. The wait happens inside the
    # block, while the session is still alive: leaving it first would stop the
    # pending runs by killing the session, which is not the mechanism here.
    starts = np.tile(initial_values, (5, 1))
    with threads(workers=2):
        with pytest.raises(ValueError, match="boom"):
            optimize_many(config, starts, boom, limit=1)
        assert not ran_again.wait(timeout=0.2)
        with lock:
            assert calls == 1


@pytest.mark.timeout(60)
def test_optimize_many_leaves_the_block_usable_after_a_failure(
    config: Any, test_functions: Any
) -> None:
    # Every run reaches its first evaluation before any of them returns, so the
    # three siblings are provably in flight when the fourth fails and are
    # abandoned rather than skipped. A barrier gives that ordering outright;
    # sleeping for it only makes it likely. One worker per run, so the
    # rendezvous cannot starve on the pool.
    started = threading.Barrier(4)

    def boom(_v: Any, _c: Any) -> float:
        started.wait(timeout=30)
        msg = "boom"
        raise ValueError(msg)

    def first_evaluation_waits() -> Any:
        waited = False

        def objective(variables: Any, context: Any) -> float:
            nonlocal waited
            if not waited:
                waited = True
                started.wait(timeout=30)
            return float(test_functions[0](variables, context))

        return objective

    starts = np.tile(initial_values, (4, 1))
    with threads(workers=4):
        with pytest.raises(ValueError, match="boom"):
            optimize_many(
                config,
                starts,
                [
                    first_evaluation_waits(),
                    first_evaluation_waits(),
                    boom,
                    first_evaluation_waits(),
                ],
            )
        # Siblings are abandoned, not cancelled, so they may still be running
        # here; the block must stay usable and must not deadlock against them.
        # The executor has to still be there: without this check a run that
        # quietly fell back to evaluating in-process would satisfy the exit
        # code just as well.
        assert compose.current_executor() is not None
        result = optimize(config, initial_values, test_functions[0])
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


@pytest.mark.timeout(60)
def test_run_abandoned_by_fail_fast_returns(config: Any, test_functions: Any) -> None:
    # `gather_shared` is the primitive `optimize_many` runs its drivers on. A
    # run already in flight when a sibling fails cannot be cancelled, so it
    # keeps going until its next evaluation finds the executor gone. It must
    # then end by *returning* a result — that is why a fail-fast failure never
    # sprays exceptions out of its driver threads.
    #
    # Which exit code it returns is a race and must not be asserted: usually
    # the session reports the stop first (EXECUTOR_STOPPED), but the optimizer
    # sometimes ends its own loop first and reports OPTIMIZER_FINISHED. Over
    # 111 abandoned runs both codes appeared and none ever raised.
    outcomes: list[Any] = []
    lock = threading.Lock()
    started = threading.Barrier(4)
    finished = threading.Semaphore(0)

    def record() -> None:
        started.wait(timeout=30)
        try:
            result = optimize(config, initial_values, test_functions[0])
        except BaseException as exc:  # ruff: ignore[blind-except]
            with lock:
                outcomes.append(exc)
        else:
            with lock:
                outcomes.append(result.exit_code)
        finally:
            finished.release()

    def boom() -> None:
        # The rendezvous puts every sibling past the point where it could be
        # skipped, so all three are genuinely abandoned rather than never run.
        started.wait(timeout=30)
        msg = "boom"
        raise ValueError(msg)

    with threads(workers=4), pytest.raises(ValueError, match="boom"):
        compose.gather_shared([record, record, boom, record], None)

    # Each abandoned run releases the semaphore as it ends, so this returns as
    # soon as the last one does; the timeout is only a ceiling on a hang.
    for _ in range(3):
        assert finished.acquire(timeout=30), "abandoned runs never finished"

    assert len(outcomes) == 3
    assert all(isinstance(outcome, ExitCode) for outcome in outcomes), outcomes
    assert set(outcomes) <= {ExitCode.EXECUTOR_STOPPED, ExitCode.OPTIMIZER_FINISHED}


def test_shared_handler_aggregates_single_run(config: Any, test_functions: Any) -> None:
    history = HistoryHandler()
    with handlers(history):
        optimize(config, initial_values, test_functions[0])
    assert history["results"]


def test_shared_handler_without_execution_manager_aggregates_runs(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    with handlers(history):
        optimize(config, initial_values, test_functions[0])
        optimize(config, initial_values, test_functions[0])
    assert history["results"]


def test_shared_handler_aggregates_across_optimize_many(
    config: Any, test_functions: Any
) -> None:
    single = HistoryHandler()
    with handlers(single):
        optimize(config, initial_values, test_functions[0])

    shared = HistoryHandler()
    starts = np.tile(initial_values, (3, 1))
    with threads(workers=2), handlers(shared):
        optimize_many(config, starts, test_functions[0])

    assert len(shared["results"]) > len(single["results"])


def test_nested_handlers_inherit_by_default(config: Any, test_functions: Any) -> None:
    outer = HistoryHandler()
    inner = HistoryHandler()
    with handlers(outer), handlers(inner):
        optimize(config, initial_values, test_functions[0])
    assert inner["results"]
    assert outer["results"]
    assert len(outer["results"]) == len(inner["results"])


def test_nested_handlers_inherit_false_isolates(
    config: Any, test_functions: Any
) -> None:
    outer = HistoryHandler()
    inner = HistoryHandler()
    with handlers(outer), handlers(inner, inherit=False):
        optimize(config, initial_values, test_functions[0])
    assert inner["results"]
    assert outer["results"] is None


def test_nested_handlers_inherit_false_with_manual_relist(
    config: Any, test_functions: Any
) -> None:
    outer = HistoryHandler()
    other = HistoryHandler()
    inner = HistoryHandler()
    with handlers(outer, other), handlers(inner, outer, inherit=False):
        optimize(config, initial_values, test_functions[0])
    assert inner["results"]
    assert outer["results"]
    assert len(outer["results"]) == len(inner["results"])
    assert other["results"] is None


def test_inherit_steals_from_all_enclosing_across_inherit_false(
    config: Any, test_functions: Any
) -> None:
    a = HistoryHandler()
    b = HistoryHandler()
    c = HistoryHandler()
    with handlers(a), handlers(b, inherit=False), handlers(c):
        optimize(config, initial_values, test_functions[0])
    assert a["results"]
    assert b["results"]
    assert c["results"]


def test_inherited_handler_returns_to_enclosing_block_after_nested_exit(
    config: Any, test_functions: Any
) -> None:
    outer = HistoryHandler()
    with handlers(outer):
        with handlers(HistoryHandler()):
            optimize(config, initial_values, test_functions[0])
        inherited = len(outer["results"])
        optimize(config, initial_values, test_functions[0])
    assert len(outer["results"]) > inherited


def test_compose_accessors_reflect_open_blocks() -> None:
    assert compose.current_handlers() is None
    assert compose.current_executor() is None

    outer = HistoryHandler()
    inner = HistoryHandler()
    with threads(workers=1):
        assert compose.current_executor() is not None
        assert compose.current_handlers() is None
        with handlers(outer):
            outer_scope = compose.current_handlers()
            assert outer_scope is not None
            with handlers(inner):
                assert compose.current_handlers() is not outer_scope
            assert compose.current_handlers() is outer_scope
    assert compose.current_handlers() is None
    assert compose.current_executor() is None


if _TEST_HPC:

    class _MockedHPCAdapter:
        def __init__(self, path: Path) -> None:
            self._path = path
            self._jobs: dict[int, str] = {}
            self._job_id = 0

        def submit_job(self, job_name: str, command: str, **_kwargs: Any) -> int:
            *_, input_file, output_file = command.split()
            threading.Thread(
                target=run_task, args=(input_file, output_file), daemon=True
            ).start()
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

        def get_status_of_my_jobs(self) -> pd.DataFrame:
            running = [
                job_id
                for job_id, job_name in self._jobs.items()
                if not (self._path / f"{job_name}.out").exists()
            ]
            self._jobs = {job_id: self._jobs[job_id] for job_id in running}
            return pd.DataFrame(list(self._jobs.keys()), columns=["jobid"])


@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_evaluates_through_the_simple_api(
    config: Any, test_functions: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    with hpc(workers=2, workdir=tmp_path, template=""):
        result = evaluate(config, initial_values, test_functions[0])
    assert result.target_objective is not None
