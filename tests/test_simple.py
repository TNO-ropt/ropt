"""Tests for the sequential high-level ``optimize`` API."""

from __future__ import annotations

import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.enums import ExitCode
from ropt.exceptions import WorkflowError
from ropt.simple import (
    EvaluateResult,
    EvaluationFunctionContext,
    HistoryHandler,
    OptimizeResult,
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
from ropt.simple._session import current_executor, current_session, make_task_namer

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


def test_that_threaded_handlers_are_attached_to_run_in_a_thread() -> None:
    loop_handler = HistoryHandler()
    io_handler = HistoryHandler()
    scope = handlers(loop_handler, threaded=io_handler)
    with scope:
        in_thread = dict(scope.dispatcher._handlers)  # ruff: ignore[private-member-access]
    assert in_thread[loop_handler] is False
    assert in_thread[io_handler] is True


def test_that_threaded_accepts_a_sequence_of_handlers() -> None:
    first = HistoryHandler()
    second = HistoryHandler()
    scope = handlers(threaded=[first, second])
    with scope:
        in_thread = dict(scope.dispatcher._handlers)  # ruff: ignore[private-member-access]
    assert in_thread[first] is True
    assert in_thread[second] is True


def test_that_a_threaded_handler_keeps_its_flag_when_inherited() -> None:
    io_handler = HistoryHandler()
    outer = handlers(threaded=io_handler)
    with outer:
        inner = handlers()
        with inner:
            assert dict(inner.dispatcher._handlers)[io_handler] is True  # ruff: ignore[private-member-access]
        assert dict(outer.dispatcher._handlers)[io_handler] is True  # ruff: ignore[private-member-access]


def test_that_relisting_a_handler_overrides_its_threaded_flag() -> None:
    handler = HistoryHandler()
    outer = handlers(threaded=handler)
    with outer:
        inner = handlers(handler)  # re-listed as a loop-thread handler
        with inner:
            assert dict(inner.dispatcher._handlers)[handler] is False  # ruff: ignore[private-member-access]
        assert dict(outer.dispatcher._handlers)[handler] is True  # ruff: ignore[private-member-access]


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
    assert history.claimed is False


def test_optimize_local_handler_released_after_error(config: Any) -> None:
    history = HistoryHandler()

    def _boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        optimize(config, initial_values, _boom, handlers=[history])
    assert history.claimed is False


def test_optimize_local_handler_claimed_during_run(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    observed: list[bool] = []

    def _report(_: EvaluateResult) -> None:
        observed.append(history.claimed)
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
    assert history.claimed is False


def test_optimize_local_handler_claim_rolls_back_on_failure(
    config: Any, test_functions: Any
) -> None:
    first = HistoryHandler()
    second = HistoryHandler()
    second.claim()  # stands in for a handler already in use by another run
    with pytest.raises(WorkflowError, match="already been claimed"):
        optimize(config, initial_values, test_functions[0], handlers=[first, second])
    assert first.claimed is False


def test_that_report_callback_returning_true_stops_optimization(
    config: Any, test_functions: Any
) -> None:
    reported = 0

    def _report(_: EvaluateResult) -> bool:
        nonlocal reported
        reported += 1
        return True

    result = optimize(config, initial_values, test_functions[0], report=_report)
    assert result.exit_code == ExitCode.USER_ABORT
    assert reported == 1


def test_that_report_callback_stops_only_its_own_run_in_optimize_many(
    config: Any, test_functions: Any
) -> None:
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


def test_task_name_omits_perturbation_index_for_unperturbed_evaluation() -> None:
    with threads(workers=1):
        namer = make_task_namer(current_session(), current_executor())
        assert namer is not None
        context = EvaluationFunctionContext(
            realization=2, perturbation=-1, batch_id=3, eval_idx=0
        )
        assert namer([context]) == "run0-b3-r2"


def test_task_name_includes_perturbation_index_for_perturbed_evaluation() -> None:
    with threads(workers=1):
        namer = make_task_namer(current_session(), current_executor())
        assert namer is not None
        context = EvaluationFunctionContext(
            realization=2, perturbation=5, batch_id=3, eval_idx=0
        )
        assert namer([context]) == "run0-b3-r2-p5"


def test_task_name_uses_the_first_context_of_a_bundle() -> None:
    with threads(workers=1):
        namer = make_task_namer(current_session(), current_executor())
        assert namer is not None
        contexts = [
            EvaluationFunctionContext(
                realization=4, perturbation=-1, batch_id=1, eval_idx=0
            ),
            EvaluationFunctionContext(
                realization=5, perturbation=-1, batch_id=1, eval_idx=1
            ),
        ]
        assert namer(contexts) == "run0-b1-r4"


def test_task_run_ids_are_unique_within_an_execution_block() -> None:
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, batch_id=0, eval_idx=0
    )
    with threads(workers=1):
        first = make_task_namer(current_session(), current_executor())
        second = make_task_namer(current_session(), current_executor())
        assert first is not None
        assert second is not None
        assert first([context]) == "run0-b0-r0"
        assert second([context]) == "run1-b0-r0"


def test_task_run_ids_restart_for_each_execution_block() -> None:
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, batch_id=0, eval_idx=0
    )
    with threads(workers=1):
        first = make_task_namer(current_session(), current_executor())
    with threads(workers=1):
        second = make_task_namer(current_session(), current_executor())
    assert first is not None
    assert second is not None
    assert first([context]) == "run0-b0-r0"
    assert second([context]) == "run0-b0-r0"


def test_no_task_namer_without_an_execution_block() -> None:
    assert make_task_namer(current_session(), current_executor()) is None


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
    # The outer evaluation opens its own independent block and optimizer.
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
    # For a fixed outer value, minimized at b = 3, leaving (outer_value - 2) ** 2.
    b = float(variables[0])
    return (outer_value - 2.0) ** 2 + (b - 3.0) ** 2


def _bilevel_outer(variables: NDArray[np.float64], _context: Any) -> float:
    # Each outer evaluation runs an isolated inner optimization over b.
    a = float(variables[0])
    with threads(workers=1):
        inner = optimize(
            _BILEVEL_CONFIG, [0.0], partial(_inner_objective, outer_value=a)
        )
    assert inner.target_objective is not None
    return inner.target_objective


def test_isolated_nested_optimization_on_threads() -> None:
    # A small bilevel problem: the outer converges to a = 2 while every inner
    # run (isolated, on its own threads block) drives b to 3.
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
    # gather_shared sets the block's session on each driver thread, so a
    # session-consuming call (offload here) resolves the shared executor there.
    def _square(value: int) -> int:
        return value * value

    with threads(workers=2):
        session = current_session()
        assert session is not None
        functions = [partial(_square, i) for i in (1, 2, 3)]
        [result] = session.gather_shared([lambda: offload(functions)], limit=1)
    assert result == (1, 4, 9)


def _double(value: float) -> float:
    return 2.0 * value


def _offload_in_own_block(variables: NDArray[np.float64], _context: Any) -> float:
    # The evaluation offloads to a block it opens itself, not the outer one.
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


def test_sequential_execution_managers_are_allowed(
    config: Any, test_functions: Any
) -> None:
    with threads(workers=2):
        first = optimize(config, initial_values, test_functions[0])
    with processes(workers=2):
        second = optimize(config, initial_values, test_functions[0])
    assert first.exit_code == second.exit_code
    assert first.variables == pytest.approx(second.variables)


def test_session_clears_after_block(config: Any, test_functions: Any) -> None:
    with threads(workers=1):
        pass
    # A sequential run works again once the block has exited.
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
    assert compose.current_session() is None
    assert compose.current_handlers() is None
    assert compose.current_executor() is None

    outer = HistoryHandler()
    inner = HistoryHandler()
    with threads(workers=1):
        assert compose.current_session() is not None
        assert compose.current_executor() is not None
        assert compose.current_handlers() is None
        with handlers(outer):
            outer_scope = compose.current_handlers()
            assert outer_scope is not None
            with handlers(inner):
                assert compose.current_handlers() is not outer_scope
            assert compose.current_handlers() is outer_scope
    assert compose.current_session() is None
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
