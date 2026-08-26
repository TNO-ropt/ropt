"""Tests for the sequential high-level ``optimize`` API."""

# The monkeypatched tests here name their target as an attribute and assert it
# was used. An earlier version patched a method by string and became a no-op the
# day it was renamed, which mypy cannot see. The other traps in this file are
# explained where they sit.

from __future__ import annotations

import os
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.compute_steps import EvaluationStep, OptimizationStep
from ropt.components.concurrency import run_concurrent
from ropt.components.evaluators import FunctionEvaluator
from ropt.components.event_handlers import EventDispatcher
from ropt.components.executors import ThreadingExecutor
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import ExecutorStopped, WorkflowError
from ropt.simple import (
    EvaluateResult,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    HistoryHandler,
    OptimizeResult,
    SharedHandlers,
    WorkerPool,
    evaluate,
    evaluate_many,
    offload,
    optimize,
    optimize_many,
    session,
)
from ropt.simple._function import adapt_function
from ropt.simple._session import _Session

if TYPE_CHECKING:
    from collections.abc import Callable
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


def test_report_callback_receives_evaluated_variables(
    config: Any, test_functions: Any
) -> None:
    # The optimizer chooses these points, so the caller has no other way to
    # learn which one an evaluation belongs to.
    reported: list[EvaluateResult] = []
    optimize(config, initial_values, test_functions[0], report=reported.append)
    variables = [item.variables for item in reported if item.variables is not None]
    assert variables
    assert len(variables) == len(reported)
    assert all(item.shape == initial_values.shape for item in variables)
    assert any(not np.array_equal(item, initial_values) for item in variables)


def test_optimize_result_is_an_evaluate_result(
    config: Any, test_functions: Any
) -> None:
    # A run ends at one evaluation, so its result carries that evaluation's
    # fields and adds only the exit code.
    result = optimize(config, initial_values, test_functions[0])
    assert isinstance(result, EvaluateResult)
    assert result.variables is not None


def test_group_report_callback_reports_across_runs(
    config: Any, test_functions: Any
) -> None:
    reported: list[EvaluateResult] = []
    with session() as active:
        group = active.shared_handlers(report=reported.append)
        optimize(config, initial_values, test_functions[0], handlers=[group])
        after_first = len(reported)
        optimize(config, initial_values, test_functions[0], handlers=[group])
    assert after_first > 0
    assert len(reported) > after_first
    assert all(isinstance(item, EvaluateResult) for item in reported)


def test_group_threaded_handlers_run_in_thread() -> None:
    loop_handler = HistoryHandler()
    io_handler = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(loop_handler, threaded=io_handler)
        in_thread = dict(group._dispatcher._handlers)  # ruff: ignore[private-member-access]
    assert in_thread[loop_handler] is False
    assert in_thread[io_handler] is True


def test_group_threaded_accepts_handler_sequence() -> None:
    first = HistoryHandler()
    second = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(threaded=[first, second])
        in_thread = dict(group._dispatcher._handlers)  # ruff: ignore[private-member-access]
    assert in_thread[first] is True
    assert in_thread[second] is True


def test_empty_group_adds_no_forwarding_handler(
    config: Any, test_functions: Any
) -> None:
    with session() as active:
        group = active.shared_handlers()
        step = EvaluationStep(
            evaluator=FunctionEvaluator(
                function=adapt_function(test_functions[0], 1, 0)
            )
        )
        # Nothing wants events, so the run must not be given a forwarding handler.
        group.attach_to(step)
        assert step.event_handlers == []
        result = optimize(config, initial_values, test_functions[0])
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


def test_hand_assembled_step_runs(config: Any, test_functions: Any) -> None:
    # A step built by hand runs exactly the way optimize() runs its own: it
    # takes its context and nothing from its surroundings.
    step = OptimizationStep(
        evaluator=FunctionEvaluator(function=adapt_function(test_functions[0], 1, 0))
    )
    history = HistoryHandler()
    step.add_event_handler(history)
    exit_code = step.run(
        context=EnOptContext.model_validate(config), variables=initial_values
    )
    assert exit_code == ExitCode.OPTIMIZER_FINISHED
    assert len(history["results"]) > 1


def test_handler_reusable_after_group_closes() -> None:
    handler = HistoryHandler()
    with session() as active:
        active.shared_handlers(handler)
    # A new session, since a group cannot be reopened once its own has closed.
    with session() as active:
        group = active.shared_handlers(handler)
        assert set(dict(group._dispatcher._handlers)) == {handler}  # ruff: ignore[private-member-access]


def test_handler_reusable_after_group_fails_to_open() -> None:
    good = HistoryHandler()
    bad = HistoryHandler()
    with session() as active:
        with pytest.raises(WorkflowError, match="already registered with a dispatcher"):
            active.shared_handlers(good, bad, bad)
        group = active.shared_handlers(good, bad)
        assert set(dict(group._dispatcher._handlers)) == {good, bad}  # ruff: ignore[private-member-access]


def test_local_handler_refused_by_group(config: Any, test_functions: Any) -> None:
    handler = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[handler])
    with (
        session() as active,
        pytest.raises(WorkflowError, match="already in use") as excinfo,
    ):
        active.shared_handlers(handler)
    # The low-level refusal names a compute step, which this API never hands
    # out; a reader cannot act on it.
    assert "compute step" not in str(excinfo.value)
    assert "separate handler" in str(excinfo.value)


def test_handler_in_open_group_refused_by_another_group() -> None:
    handler = HistoryHandler()
    with session() as active:
        active.shared_handlers(handler)
        with pytest.raises(WorkflowError, match="already in use") as excinfo:
            active.shared_handlers(handler)
    assert "compute step" not in str(excinfo.value)


def test_shared_handlers_releases_on_rollback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _failing_remove(*_args: object, **_kwargs: object) -> None:
        msg = "rollback failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(EventDispatcher, "remove_event_handler", _failing_remove)
    handler = HistoryHandler()
    with (
        session() as active,
        pytest.raises(RuntimeError, match="rollback failed"),
    ):
        active.shared_handlers(handler, handler)


def test_group_close_is_idempotent() -> None:
    with session() as active:
        group = active.shared_handlers(HistoryHandler())
        group.close()
        group.close()


def test_group_context_manager_closes_the_group() -> None:
    handler = HistoryHandler()
    with session() as active:
        with active.shared_handlers(handler) as group:
            assert set(dict(group._dispatcher._handlers)) == {handler}  # ruff: ignore[private-member-access]
        # Released, so a plain dispatcher can claim it again.
        EventDispatcher().add_event_handler(handler)


def test_group_handler_reusable_as_local_after_close(
    config: Any, test_functions: Any
) -> None:
    handler = HistoryHandler()
    with session() as active:
        active.shared_handlers(handler).close()
    optimize(config, initial_values, test_functions[0], handlers=[handler])
    assert len(handler["results"]) > 0


def test_optimize_mixes_local_handler_and_group(
    config: Any, test_functions: Any
) -> None:
    local = HistoryHandler()
    shared = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(shared)
        optimize(config, initial_values, test_functions[0], handlers=[local, group])
    assert local["results"]
    assert shared["results"]
    assert local._claimed is False  # ruff: ignore[private-member-access]


def test_optimize_feeds_two_groups_at_once(config: Any, test_functions: Any) -> None:
    first = HistoryHandler()
    second = HistoryHandler()
    with session() as active:
        group_a = active.shared_handlers(first)
        group_b = active.shared_handlers(second)
        optimize(config, initial_values, test_functions[0], handlers=[group_a, group_b])
    assert first["results"]
    assert second["results"]
    assert len(first["results"]) == len(second["results"])


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
    with session() as active:
        results = optimize_many(
            config,
            x0,
            test_functions[0],
            report=[_stop, _continue],
            pool=active.thread_pool(workers=2),
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
    assert result.results is not None
    assert result.results.evaluations.variables.shape == (initial_values.size,)


def test_evaluate_reports_the_evaluated_point(config: Any, test_functions: Any) -> None:
    result = evaluate(config, initial_values, test_functions[0])
    assert result.variables is not None
    assert np.array_equal(result.variables, initial_values)


def test_thread_run_sees_only_the_handlers_it_is_given(
    config: Any, test_functions: Any
) -> None:
    handler = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(handler)

        def _without_handlers() -> None:
            optimize(config, initial_values, test_functions[0])

        thread = threading.Thread(target=_without_handlers)
        thread.start()
        thread.join()
        assert handler["results"] is None

        def _with_handlers() -> None:
            optimize(config, initial_values, test_functions[0], handlers=[group])

        thread = threading.Thread(target=_with_handlers)
        thread.start()
        thread.join()
    assert handler["results"] is not None


def test_evaluate_feeds_a_shared_group(config: Any, test_functions: Any) -> None:
    handler = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(handler)
        evaluate(config, initial_values, test_functions[0], handlers=[group])
        evaluate(
            config, np.zeros(initial_values.size), test_functions[0], handlers=[group]
        )
    assert len(handler["results"]) == 2


def test_evaluate_accepts_a_local_handler(config: Any, test_functions: Any) -> None:
    history = HistoryHandler()
    evaluate(config, initial_values, test_functions[0], handlers=[history])
    assert len(history["results"]) == 1
    assert history._claimed is False  # ruff: ignore[private-member-access]


def test_evaluate_many_accepts_a_local_handler(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    evaluate_many(config, matrix, test_functions[0], handlers=[history])
    assert len(history["results"]) == 2


def test_evaluate_report_return_value_ignored(config: Any, test_functions: Any) -> None:
    reported: list[EvaluateResult] = []

    def _stop(result: EvaluateResult) -> bool:
        reported.append(result)
        return True

    result = evaluate(config, initial_values, test_functions[0], report=_stop)
    assert len(reported) == 1
    assert result.target_objective == pytest.approx(0.66)


def test_evaluate_many_report_return_value_ignored(
    config: Any, test_functions: Any
) -> None:
    # The callback returns True on the very first result, which stops the
    # forwarding of further results to it -- but the batch itself already ran
    # to completion before the event fired, so every row still comes back.
    reported: list[EvaluateResult] = []

    def _stop(result: EvaluateResult) -> bool:
        reported.append(result)
        return True

    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    results = evaluate_many(config, matrix, test_functions[0], report=_stop)
    assert len(reported) == 1
    assert len(results) == 2


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


def test_optimize_with_thread_pool(config: Any, test_functions: Any) -> None:
    with session() as active:
        result = optimize(
            config,
            initial_values,
            test_functions[0],
            pool=active.thread_pool(workers=2),
        )
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


def test_optimize_evaluates_on_the_given_pool(config: Any) -> None:
    # A pool is only ever what a run is handed; proving the evaluation lands on
    # one of its worker threads, not the caller's, is what distinguishes a run
    # that really dispatches from one that silently fell back to in-process.
    seen: list[str] = []
    lock = threading.Lock()

    def _record_thread(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        with lock:
            seen.append(threading.current_thread().name)
        return 0.0

    with session() as active:
        optimize(
            config, initial_values, _record_thread, pool=active.thread_pool(workers=2)
        )
    assert seen
    assert threading.current_thread().name not in seen


def test_optimize_without_a_pool_evaluates_in_process(config: Any) -> None:
    # The mirror of the above: with no pool passed, a run must evaluate on the
    # calling thread rather than reaching for some pool from its surroundings.
    seen: list[str] = []
    lock = threading.Lock()

    def _record_thread(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        with lock:
            seen.append(threading.current_thread().name)
        return 0.0

    optimize(config, initial_values, _record_thread)
    assert seen
    assert set(seen) == {threading.current_thread().name}


def test_batch_ids_are_unique_across_runs_sharing_a_pool(config: Any) -> None:
    first: list[int] = []
    second: list[int] = []
    lock = threading.Lock()
    with session() as active:
        pool = active.thread_pool(workers=2)
        optimize(config, initial_values, _collect_batch_ids(first, lock), pool=pool)
        optimize(config, initial_values, _collect_batch_ids(second, lock), pool=pool)
    assert first
    assert second
    assert not set(first) & set(second)


def test_concurrent_optimize_calls_sharing_a_pool_get_unique_batch_ids(
    config: Any,
) -> None:
    first: list[int] = []
    second: list[int] = []
    lock = threading.Lock()
    with session() as active:
        pool = active.thread_pool(workers=4)

        def _run(sink: list[int]) -> None:
            optimize(config, initial_values, _collect_batch_ids(sink, lock), pool=pool)

        driver_a = threading.Thread(target=_run, args=(first,))
        driver_b = threading.Thread(target=_run, args=(second,))
        driver_a.start()
        driver_b.start()
        driver_a.join()
        driver_b.join()
    assert first
    assert second
    assert not set(first) & set(second)


def test_batch_ids_are_unique_across_concurrent_runs(config: Any) -> None:
    sinks: list[list[int]] = [[], [], []]
    lock = threading.Lock()
    with session() as active:
        optimize_many(
            config,
            initial_values,
            [_collect_batch_ids(sink, lock) for sink in sinks],
            pool=active.thread_pool(workers=2),
        )
    for sink in sinks:
        assert sink
    assert sum(len(set(sink)) for sink in sinks) == len(set().union(*sinks))


def test_optimize_many_without_a_pool_gets_unique_batch_ids(config: Any) -> None:
    # optimize_many builds one private serial pool for the whole call when
    # given none, shared by every concurrent run, so their batch IDs still
    # stay apart even though nothing here ever dispatches to a worker.
    sinks: list[list[int]] = [[], [], []]
    lock = threading.Lock()
    optimize_many(
        config,
        initial_values,
        [_collect_batch_ids(sink, lock) for sink in sinks],
    )
    for sink in sinks:
        assert sink
    assert sum(len(set(sink)) for sink in sinks) == len(set().union(*sinks))


def test_batch_ids_restart_for_each_pool(config: Any) -> None:
    first: list[int] = []
    second: list[int] = []
    lock = threading.Lock()
    with session() as active:
        optimize(
            config,
            initial_values,
            _collect_batch_ids(first, lock),
            pool=active.thread_pool(workers=2),
        )
        optimize(
            config,
            initial_values,
            _collect_batch_ids(second, lock),
            pool=active.thread_pool(workers=2),
        )
    assert min(first) == 0
    assert min(second) == 0


def test_batch_ids_restart_for_each_run_without_a_pool(
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


def test_metadata_reaches_the_evaluation_function_with_a_pool(config: Any) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    with session() as active:
        optimize(
            config,
            initial_values,
            _record_metadata(seen, lock),
            metadata={"run": 7},
            pool=active.thread_pool(workers=2),
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
    with session() as active:
        optimize_many(
            config,
            initial_values,
            [_record_metadata(first, lock), _record_metadata(second, lock)],
            metadata=[{"run": 0}, {"run": 1}],
            pool=active.thread_pool(workers=2),
        )
    assert all(item == {"run": 0} for item in first)
    assert all(item == {"run": 1} for item in second)


def test_evaluate_many_with_a_thread_pool(config: Any, test_functions: Any) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    with session() as active:
        results = evaluate_many(
            config, matrix, test_functions[0], pool=active.thread_pool(workers=2)
        )
    for result, expected in zip(results, [0.66, 0.75], strict=True):
        assert result.target_objective == pytest.approx(expected)


def test_evaluate_with_a_thread_pool(config: Any, test_functions: Any) -> None:
    with session() as active:
        result = evaluate(
            config,
            initial_values,
            test_functions[0],
            pool=active.thread_pool(workers=2),
        )
    assert result.target_objective == pytest.approx(0.66)


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
    # A run's evaluation function is plain code: it may open a session and a
    # pool of its own, nested inside whatever pool is running it. Nothing
    # ambient needs to be threaded through for that to work.
    with session() as active:
        result = optimize(
            _INNER_CONFIG, variables, _sphere, pool=active.thread_pool(workers=1)
        )
    assert result.target_objective is not None
    return result.target_objective


def _offload_from_evaluation(_variables: NDArray[np.float64], _context: Any) -> float:
    return float(offload(_return_one))


def _return_one() -> float:
    return 1.0


def test_evaluation_function_can_open_its_own_thread_pool(config: Any) -> None:
    with session() as active:
        result = optimize(
            config,
            initial_values,
            _run_inner_optimization,
            pool=active.thread_pool(workers=2),
        )
    assert result.variables is not None
    assert result.target_objective is not None


@pytest.mark.slow
def test_evaluation_function_can_open_its_own_process_pool(config: Any) -> None:
    with session() as active:
        result = optimize(
            config,
            initial_values,
            _run_inner_optimization,
            pool=active.process_pool(workers=2),
        )
    assert result.variables is not None
    assert result.target_objective is not None


_BUNDLE_CONFIG: dict[str, Any] = {
    "variables": {"variable_count": 2},
    "realizations": {"weights": [1.0] * 4},
}
_BUNDLE_THREADS: set[int] = set()


def _record_thread(
    variables: NDArray[np.float64], _context: EvaluationFunctionContext
) -> float:
    _BUNDLE_THREADS.add(threading.get_ident())
    return float(np.sum(variables**2))


def test_negative_bundle_size_refused() -> None:
    # A serial pool builds no evaluator, so nothing downstream would catch it.
    with pytest.raises(ValueError, match="bundle_size must be >= 0"):
        WorkerPool(bundle_size=-1)


@pytest.mark.timeout(30)
def test_bundle_size_sends_the_whole_batch_as_one_task() -> None:
    # The evaluations in one task run after each other, so a whole-batch bundle
    # is observable as a single worker doing all four realizations.
    with session() as active:
        pool = active.thread_pool(workers=4, bundle_size=0)
        assert pool.bundle_size == 0
        _BUNDLE_THREADS.clear()
        evaluate(_BUNDLE_CONFIG, np.zeros(2), _record_thread, pool=pool)
    assert len(_BUNDLE_THREADS) == 1


_NESTED_INNER: dict[str, Any] = {
    "variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6},
    "optimizer": {"max_functions": 2},
}
_NESTED_OUTER: dict[str, Any] = {"variables": {"variable_count": 2}}
# Two points, one batch: exactly two outer evaluations, so the barrier party is
# known rather than dependent on how the optimizer schedules its batches.
_NESTED_POINTS = np.array([[0.5, 0.5], [1.5, 1.5]])


def _pid_sphere(
    variables: NDArray[np.float64], _context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(
        objectives=float(np.sum(variables**2)), metadata={"pid": os.getpid()}
    )


def _nested_run(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    *,
    pool: WorkerPool,
    group: SharedHandlers,
    barrier: threading.Barrier,
) -> float:
    # Neither outer evaluation can pass until both have arrived, so if they were
    # run one after the other this breaks the barrier instead of quietly passing.
    barrier.wait()
    result = optimize(
        _NESTED_INNER,
        variables,
        _pid_sphere,
        pool=pool,
        handlers=[group],
        metadata={"outer": context.eval_idx},
    )
    assert result.target_objective is not None
    return result.target_objective


@pytest.mark.parametrize(
    "processes",
    [
        pytest.param(False, id="threads"),
        pytest.param(True, id="processes", marks=pytest.mark.slow),
    ],
)
@pytest.mark.timeout(120)
def test_concurrent_inner_runs_on_a_second_pool_feed_one_group(
    processes: Any,
) -> None:
    history = HistoryHandler()
    barrier = threading.Barrier(len(_NESTED_POINTS), timeout=30)
    with session() as active:
        inner = (
            active.process_pool(workers=2)
            if processes
            else active.thread_pool(workers=2)
        )
        outer = active.thread_pool(workers=len(_NESTED_POINTS))
        group = active.shared_handlers(history)
        evaluate_many(
            _NESTED_OUTER,
            _NESTED_POINTS,
            partial(_nested_run, pool=inner, group=group, barrier=barrier),
            pool=outer,
        )

    batches: dict[int, set[int]] = {}
    pids: set[int] = set()
    for item in history["results"]:
        batches.setdefault(item.metadata["outer"], set()).add(item.batch_id)
        recorded = item.evaluations.metadata.get("pid")
        if recorded is not None:
            pids.update(int(pid) for pid in np.ravel(recorded))
    # Both inner runs reached the one group, each tagged with the outer
    # evaluation that started it.
    assert set(batches) == {0, 1}
    # They drew from the pool's single counter, so their batches never collided.
    assert not batches[0] & batches[1]
    assert pids
    if processes:
        # A process pool evaluates in workers of its own.
        assert os.getpid() not in pids
    else:
        # A thread pool evaluates here, so nesting needs no picklable function.
        assert pids == {os.getpid()}


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
    with session() as active:
        inner = optimize(
            _BILEVEL_CONFIG,
            [0.0],
            partial(_inner_objective, outer_value=a),
            pool=active.thread_pool(workers=1),
        )
    assert inner.target_objective is not None
    return inner.target_objective


def test_nested_optimization_on_a_thread_pool() -> None:
    with session() as active:
        result = optimize(
            _BILEVEL_CONFIG, [0.0], _bilevel_outer, pool=active.thread_pool(workers=1)
        )
    assert result.variables is not None
    assert result.variables[0] == pytest.approx(2.0, abs=0.05)
    assert result.target_objective == pytest.approx(0.0, abs=1e-2)


def test_offload_in_evaluation_without_a_pool_runs_inline(config: Any) -> None:
    # A pool is only ever what is passed to a call, never what is discovered:
    # `offload` inside the evaluation function is given none, so it runs
    # inline even though the run itself is evaluating on a thread pool.
    with session() as active:
        result = optimize(
            config,
            initial_values,
            _offload_from_evaluation,
            pool=active.thread_pool(workers=2),
        )
    assert result.target_objective is not None


class _FatalWork(BaseException):
    """Not an Exception, so the worker loops let it reach the task group."""


def _fatal_work() -> int:
    msg = "worker died"
    raise _FatalWork(msg)


def test_fatal_worker_error_reported_when_the_session_closes() -> None:
    def _run_block() -> None:
        with session() as active, pytest.raises(ExecutorStopped):
            offload(_fatal_work, pool=active.thread_pool(workers=1))

    with pytest.raises(BaseExceptionGroup) as excinfo:
        _run_block()
    matched, _ = excinfo.value.split(_FatalWork)
    assert matched is not None


def _double(value: float) -> float:
    return 2.0 * value


def _offload_in_own_pool(variables: NDArray[np.float64], _context: Any) -> float:
    with session() as active:
        doubled = offload(
            [partial(_double, 3.0), partial(_double, 4.0)],
            pool=active.thread_pool(workers=2),
        )
    assert doubled == (6.0, 8.0)
    return float(variables @ variables)


def _own_handlers_in_evaluation(variables: NDArray[np.float64], _context: Any) -> float:
    # A run's evaluation function is plain code: it may open a session and a
    # shared handlers group of its own, nested inside whatever pool is running it.
    history = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(history)
        optimize(_INNER_CONFIG, variables, _sphere, handlers=[group])
    assert len(history.results) > 0
    return float(variables @ variables)


def test_offload_in_evaluation_uses_its_own_pool(config: Any) -> None:
    with session() as active:
        result = optimize(
            config,
            initial_values,
            _offload_in_own_pool,
            pool=active.thread_pool(workers=2),
        )
    assert result.target_objective is not None


def test_group_in_an_evaluation(config: Any) -> None:
    with session() as active:
        result = optimize(
            config,
            initial_values,
            _own_handlers_in_evaluation,
            pool=active.thread_pool(workers=2),
        )
    assert result.target_objective is not None


@pytest.mark.slow
def test_sequential_pools_are_allowed(config: Any, test_functions: Any) -> None:
    with session() as active:
        first = optimize(
            config,
            initial_values,
            test_functions[0],
            pool=active.thread_pool(workers=2),
        )
    with session() as active:
        second = optimize(
            config,
            initial_values,
            test_functions[0],
            pool=active.process_pool(workers=2),
        )
    assert first.exit_code == second.exit_code
    assert first.variables == pytest.approx(second.variables)


def test_session_without_task_group_reports_stopped() -> None:
    sess = _Session()
    with pytest.raises(WorkflowError, match="is not running"):
        sess.open_dispatcher(EventDispatcher())
    with pytest.raises(WorkflowError, match="is not running"):
        sess.open_pool(lambda: ThreadingExecutor(workers=1))


def test_thread_pool_objective_exception_propagates(config: Any) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"), session() as active:
        optimize(config, initial_values, boom, pool=active.thread_pool(workers=2))


def test_thread_pool_survives_objective_exception(
    config: Any, test_functions: Any
) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    with session() as active:
        pool = active.thread_pool(workers=2)
        with pytest.raises(ValueError, match="boom"):
            optimize(config, initial_values, boom, pool=pool)
        # The pool survives a failed run and can still be used by the next one.
        result = optimize(config, initial_values, test_functions[0], pool=pool)
        assert result.variables is not None
        assert np.allclose(result.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_optimize_with_process_pool(config: Any, test_functions: Any) -> None:
    with session() as active:
        result = optimize(
            config,
            initial_values,
            test_functions[0],
            pool=active.process_pool(workers=2),
        )
    assert result.variables is not None
    assert np.allclose(result.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_process_pool_without_cloudpickle(config: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor._HAVE_CLOUDPICKLE",
        False,
    )
    with session() as active:
        result = optimize(
            config, initial_values, _sphere, pool=active.process_pool(workers=2)
        )
    assert result.variables is not None
    assert np.allclose(result.variables, 0.0, atol=0.02)


@pytest.mark.slow
def test_evaluate_without_cloudpickle(config: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor._HAVE_CLOUDPICKLE",
        False,
    )
    with session() as active:
        result = evaluate(
            config, initial_values, _sphere, pool=active.process_pool(workers=2)
        )
    assert result.target_objective == pytest.approx(0.01)


def test_optimize_many_broadcasts_config_and_objective(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with session() as active:
        results = optimize_many(
            config, starts, test_functions[0], pool=active.thread_pool(workers=2)
        )
    assert len(results) == 2
    assert all(isinstance(result, OptimizeResult) for result in results)
    for result in results:
        assert result.variables is not None
        assert np.allclose(result.variables, 0.5, atol=0.02)


def test_optimize_many_per_run_objectives(config: Any, test_functions: Any) -> None:
    with session() as active:
        results = optimize_many(
            config,
            initial_values,
            [test_functions[0], test_functions[1]],
            pool=active.thread_pool(workers=2),
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
    with session() as active:
        optimize_many(
            config,
            starts,
            test_functions[0],
            report=reported.append,
            pool=active.thread_pool(workers=2),
        )
    assert reported
    assert all(isinstance(item, EvaluateResult) for item in reported)


def test_optimize_many_accepts_a_report_per_run(
    config: Any, test_functions: Any
) -> None:
    first: list[EvaluateResult] = []
    second: list[EvaluateResult] = []
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with session() as active:
        optimize_many(
            config,
            starts,
            test_functions[0],
            report=[first.append, second.append],
            pool=active.thread_pool(workers=2),
        )
    assert first
    assert second
    assert all(isinstance(item, EvaluateResult) for item in (*first, *second))


def test_optimize_many_rejects_mismatched_report_sequence(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with pytest.raises(ValueError, match="number of runs"):
        optimize_many(config, starts, test_functions[0], report=[lambda _r: None])


def test_optimize_many_broadcasts_a_single_metadata_dict(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with session() as active:
        results = optimize_many(
            config,
            starts,
            test_functions[0],
            metadata={"group": "g"},
            pool=active.thread_pool(workers=2),
        )
    for result in results:
        assert result.results is not None
        assert result.results.metadata["group"] == "g"


def test_optimize_many_accepts_metadata_per_run(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with session() as active:
        results = optimize_many(
            config,
            starts,
            test_functions[0],
            metadata=[{"run_id": 0}, {"run_id": 1}],
            pool=active.thread_pool(workers=2),
        )
    assert len(results) == 2
    for idx, result in enumerate(results):
        assert result.results is not None
        assert result.results.metadata["run_id"] == idx


def test_optimize_many_rejects_mismatched_metadata_sequence(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with pytest.raises(ValueError, match="number of runs"):
        optimize_many(config, starts, test_functions[0], metadata=[{"run_id": 0}])


def test_optimize_many_respects_limit(config: Any, test_functions: Any) -> None:
    starts = np.tile(initial_values, (4, 1))
    with session() as active:
        results = optimize_many(
            config,
            starts,
            test_functions[0],
            limit=2,
            pool=active.thread_pool(workers=2),
        )
    assert len(results) == 4


def test_optimize_many_mismatched_lengths_raises(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with pytest.raises(ValueError, match="same length"):
        optimize_many(
            config, starts, [test_functions[0], test_functions[1], test_functions[0]]
        )


def test_optimize_many_rejects_more_than_two_dimensional_starts(
    config: Any, test_functions: Any
) -> None:
    starts = np.tile(initial_values, (2, 2, 1))
    with pytest.raises(ValueError, match="vector or a 2-D matrix"):
        optimize_many(config, starts, test_functions[0])


def test_optimize_many_without_runs_returns_nothing(test_functions: Any) -> None:
    assert optimize_many([], initial_values, test_functions[0]) == ()


def test_optimize_many_without_a_pool_or_session(
    config: Any, test_functions: Any
) -> None:
    # optimize_many no longer requires a session at all: without a pool it
    # builds its own private serial pool and evaluates on the driver threads.
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = optimize_many(config, starts, test_functions[0])
    assert len(results) == 2
    for result in results:
        assert result.variables is not None
        assert np.allclose(result.variables, 0.5, atol=0.02)


def test_optimize_many_fail_fast(config: Any, test_functions: Any) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    starts = np.tile(initial_values, (3, 1))
    with session() as active, pytest.raises(ValueError, match="boom"):
        optimize_many(
            config,
            starts,
            [test_functions[0], boom, test_functions[0]],
            pool=active.thread_pool(workers=2),
        )


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
    with session() as active:
        pool = active.thread_pool(workers=2)
        with pytest.raises(ValueError, match="boom"):
            optimize_many(config, starts, boom, limit=1, pool=pool)
        assert not ran_again.wait(timeout=0.2)
        with lock:
            assert calls == 1


@pytest.mark.timeout(60)
def test_optimize_many_leaves_the_pool_usable_after_a_failure(
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
    with session() as active:
        pool = active.thread_pool(workers=4)
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
                pool=pool,
            )
        # Siblings are abandoned, not cancelled, so they may still be running
        # here; the pool must stay usable and must not deadlock against them.
        assert pool.executor is not None
        assert pool.executor.is_running()
        result = optimize(config, initial_values, test_functions[0], pool=pool)
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


@pytest.mark.timeout(60)
def test_run_abandoned_by_fail_fast_returns(config: Any, test_functions: Any) -> None:
    # `run_concurrent` is the primitive `optimize_many` runs its drivers on. A
    # run already in flight when a sibling fails cannot be cancelled, so it
    # keeps going until its next evaluation finds the pool closed. It must
    # then end by *returning* a result — that is why a fail-fast failure never
    # sprays exceptions out of its driver threads.
    #
    # Which exit code it returns is a race and must not be asserted: usually
    # the pool reports the stop first (EXECUTOR_STOPPED), but the optimizer
    # sometimes ends its own loop first and reports OPTIMIZER_FINISHED. Both
    # codes have been observed; neither run ever raised.
    outcomes: list[Any] = []
    lock = threading.Lock()
    started = threading.Barrier(4)
    finished = threading.Semaphore(0)

    def first_evaluation_waits() -> Any:
        # The rendezvous sits inside the run, not on the driver thread before
        # it: reaching it proves the run is past the entry check that refuses a
        # closed pool. Waiting outside only proves the run is about to start,
        # and the closing session below can beat it there -- the run is then
        # refused with a `WorkflowError` instead of abandoned in flight, which
        # is a different case and made this test flaky.
        waited = False

        def objective(variables: Any, context: Any) -> float:
            nonlocal waited
            if not waited:
                waited = True
                started.wait(timeout=30)
            return float(test_functions[0](variables, context))

        return objective

    def record(pool: WorkerPool) -> None:
        try:
            result = optimize(
                config, initial_values, first_evaluation_waits(), pool=pool
            )
        except BaseException as exc:  # ruff: ignore[blind-except]
            with lock:
                outcomes.append(exc)
        else:
            with lock:
                outcomes.append(result.exit_code)
        finally:
            finished.release()

    def boom() -> None:
        # Every sibling is inside its first evaluation by the time this passes
        # the barrier, so all three are genuinely abandoned rather than never
        # run. One worker per run, so the rendezvous cannot starve on the pool.
        started.wait(timeout=30)
        msg = "boom"
        raise ValueError(msg)

    # The block exit (a normal one: pytest.raises absorbs the failure before
    # the session sees it) stops the pool -- which is what lets the abandoned
    # runs below return instead of hanging forever.
    with session() as active:
        pool = active.thread_pool(workers=4)
        jobs: list[Callable[[], None]] = [
            partial(record, pool),
            partial(record, pool),
            boom,
            partial(record, pool),
        ]
        with pytest.raises(ValueError, match="boom"):
            run_concurrent(jobs)

    # Each abandoned run releases the semaphore as it ends, so this returns as
    # soon as the last one does; the timeout is only a ceiling on a hang.
    for _ in range(3):
        assert finished.acquire(timeout=30), "abandoned runs never finished"

    assert len(outcomes) == 3
    assert all(isinstance(outcome, ExitCode) for outcome in outcomes), outcomes
    assert set(outcomes) <= {ExitCode.EXECUTOR_STOPPED, ExitCode.OPTIMIZER_FINISHED}


def test_shared_handler_aggregates_single_run(config: Any, test_functions: Any) -> None:
    history = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(history)
        optimize(config, initial_values, test_functions[0], handlers=[group])
    assert history["results"]


def test_shared_handler_without_a_pool_aggregates_runs(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(history)
        optimize(config, initial_values, test_functions[0], handlers=[group])
        optimize(config, initial_values, test_functions[0], handlers=[group])
    assert history["results"]


def test_shared_handler_aggregates_across_optimize_many(
    config: Any, test_functions: Any
) -> None:
    single = HistoryHandler()
    with session() as active:
        group = active.shared_handlers(single)
        optimize(config, initial_values, test_functions[0], handlers=[group])

    shared = HistoryHandler()
    starts = np.tile(initial_values, (3, 1))
    with session() as active:
        group = active.shared_handlers(shared)
        optimize_many(
            config,
            starts,
            test_functions[0],
            pool=active.thread_pool(workers=2),
            handlers=[group],
        )

    assert len(shared["results"]) > len(single["results"])


def test_optimize_many_rejects_bare_handler(config: Any, test_functions: Any) -> None:
    handler = HistoryHandler()
    with pytest.raises(WorkflowError, match="takes shared handlers only"):
        optimize_many(
            config,
            initial_values,
            test_functions[0],
            handlers=[handler],  # type: ignore[list-item]
        )


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
    with session() as active:
        result = evaluate(
            config,
            initial_values,
            test_functions[0],
            pool=active.hpc_pool(workers=2, workdir=tmp_path, template=""),
        )
    assert result.target_objective is not None
