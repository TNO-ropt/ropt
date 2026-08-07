"""Tests for the sequential high-level ``optimize`` API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.components.event_handlers import HistoryHandler
from ropt.enums import ExitCode
from ropt.simple import (
    EvaluateResult,
    OptimizeResult,
    compose,
    evaluate,
    evaluate_many,
    handlers,
    optimize,
    optimize_many,
    processes,
    threads,
)
from ropt.simple._objective import adapt_objective

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    config: Any, objective: Any, test_functions: Any
) -> None:
    result = optimize(config, initial_values, objective([test_functions[0]]))
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


def test_optimize_local_handler_collects_results(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[history])
    assert history["results"] is not None
    assert len(history["results"]) > 0


def test_optimize_local_handler_rejects_reuse(config: Any, test_functions: Any) -> None:
    history = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[history])
    with pytest.raises(RuntimeError, match="already been claimed for exclusive use"):
        optimize(config, initial_values, test_functions[0], handlers=[history])


def test_adapt_objective_rejects_scalar_for_multiple_objectives() -> None:
    callback = adapt_objective(lambda _v, _c: 1.0, n_obj=2, n_con=0)
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, batch_id=0, eval_idx=0
    )
    with pytest.raises(ValueError, match="scalar objective result"):
        callback(np.zeros(2), context)


def test_adapt_objective_rejects_wrong_shape() -> None:
    callback = adapt_objective(lambda _v, _c: [1.0, 2.0, 3.0], n_obj=1, n_con=1)
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, batch_id=0, eval_idx=0
    )
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        callback(np.zeros(2), context)


def test_adapt_objective_splits_objectives_and_constraints() -> None:
    callback = adapt_objective(lambda _v, _c: [1.0, 2.0, 3.0], n_obj=1, n_con=2)
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
    assert result.exit_code == ExitCode.ENSEMBLE_EVALUATOR_FINISHED
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


def test_evaluate_multiple_objectives(config: Any, objective: Any) -> None:
    config["objectives"] = {"weights": [0.75, 0.25]}
    result = evaluate(config, initial_values, objective())
    assert result.objectives is not None
    assert result.objectives.shape == (2,)
    assert result.constraints is None


def test_optimize_with_threads(config: Any, test_functions: Any) -> None:
    with threads(workers=2):
        result = optimize(config, initial_values, test_functions[0])
    assert result.variables is not None
    assert np.allclose(result.variables, 0.5, atol=0.02)


def test_evaluate_many_with_threads(config: Any, test_functions: Any) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    with threads(workers=2):
        results = evaluate_many(config, matrix, test_functions[0])
    for result, expected in zip(results, [0.66, 0.75], strict=True):
        assert result.target_objective == pytest.approx(expected)


def test_nested_execution_managers_raise() -> None:
    with (
        threads(workers=1),
        pytest.raises(RuntimeError, match="Only one executor may be active at a time"),
        threads(workers=1),
    ):
        pass


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
    with pytest.raises(RuntimeError, match="requires an execution block"):
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


def test_nested_handlers_isolated_by_default(config: Any, test_functions: Any) -> None:
    outer = HistoryHandler()
    inner = HistoryHandler()
    with handlers(outer), handlers(inner):
        optimize(config, initial_values, test_functions[0])
    assert inner["results"]
    assert outer["results"] is None


def test_nested_handlers_relisted_enclosing_also_receives(
    config: Any, test_functions: Any
) -> None:
    outer = HistoryHandler()
    other = HistoryHandler()
    inner = HistoryHandler()
    with handlers(outer, other), handlers(inner, outer):
        optimize(config, initial_values, test_functions[0])
    assert inner["results"]
    assert outer["results"]
    assert len(outer["results"]) == len(inner["results"])
    assert other["results"] is None


def test_relisted_handler_returns_to_enclosing_block_after_nested_exit(
    config: Any, test_functions: Any
) -> None:
    outer = HistoryHandler()
    with handlers(outer):
        with handlers(outer):
            optimize(config, initial_values, test_functions[0])
        migrated_back = len(outer["results"])
        optimize(config, initial_values, test_functions[0])
    assert len(outer["results"]) > migrated_back


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
