"""Tests for the sequential high-level ``optimize`` API."""

# The monkeypatched tests here name their target as an attribute and assert it
# was used. An earlier version patched a method by string and became a no-op the
# day it was renamed, which mypy cannot see. The other traps in this file are
# explained where they sit.

from __future__ import annotations

import os
import pickle  # ruff: ignore[suspicious-pickle-import]
import threading
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.compute_steps import OptimizationStep
from ropt.components.concurrency import run_concurrent
from ropt.components.evaluators import FunctionEvaluator
from ropt.components.event_handlers import EventHandler
from ropt.components.executors import (
    HPCExecutor,
    LocalJobExecutor,
    ProcessExecutor,
    ThreadExecutor,
)
from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, ExitCode
from ropt.exceptions import ExecutionError, WorkflowError
from ropt.results import FunctionResults
from ropt.simple import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    HistoryHandler,
    OptimizationResult,
    evaluate,
    evaluate_many,
    offload,
    optimize,
    optimize_many,
)
from ropt.simple._function import adapt_function

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from ropt.components.executors import Executor
    from ropt.events import EnOptEvent

try:
    # The job path needs no extras of its own, so these tests run either way.
    import pysqa  # ruff: ignore[unused-import]

    from ropt.components.executors.__main__ import run_task

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False

initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="executors")
def executors_fixture() -> Iterator[Callable[..., Executor]]:
    """Build executors that are closed when the test ends.

    Yields:
        A factory taking an executor class (`ThreadExecutor` by default) and its
        keyword arguments.
    """
    with ExitStack() as stack:

        def _make(kind: type[Executor] = ThreadExecutor, **kwargs: Any) -> Executor:
            return stack.enter_context(kind(**kwargs))

        yield _make


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
    assert isinstance(result, OptimizationResult)
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.5, atol=0.02)
    assert result.results.target_objective == pytest.approx(0.0, abs=1e-3)
    assert result.results.functions is not None
    assert result.results.functions.objectives.shape == (1,)
    assert result.results.functions.constraints is None


def test_optimize_accepts_evaluation_function_result(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    result = optimize(config, initial_values, eval_func([test_functions[0]]))
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.5, atol=0.02)


def test_optimize_accepts_sequence_for_multiple_objectives(
    config: Any, test_functions: Any
) -> None:
    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> list[float]:
        return [func(variables, context) for func in test_functions]

    config["objectives"] = {"weights": [0.75, 0.25]}
    result = optimize(config, initial_values, objective)
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optimize_no_valid_result_has_no_results(config: Any) -> None:
    result = optimize(config, initial_values, lambda _v, _c: np.nan)
    assert result.exit_code == ExitCode.TOO_FEW_REALIZATIONS
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
    reported: list[FunctionResults] = []
    optimize(config, initial_values, test_functions[0], report=reported.append)
    assert reported
    assert all(isinstance(item, FunctionResults) for item in reported)
    assert any(item.target_objective is not None for item in reported)


def test_report_callback_receives_evaluated_variables(
    config: Any, test_functions: Any
) -> None:
    # The optimizer chooses these points, so the caller has no other way to
    # learn which one an evaluation belongs to.
    reported: list[FunctionResults] = []
    optimize(config, initial_values, test_functions[0], report=reported.append)
    variables = [item.variables for item in reported if item.variables is not None]
    assert variables
    assert len(variables) == len(reported)
    assert all(item.shape == initial_values.shape for item in variables)
    assert any(not np.array_equal(item, initial_values) for item in variables)


def test_optimize_result_carries_the_best_evaluation(
    config: Any, test_functions: Any
) -> None:
    # A run ends at one evaluation, which it hands back unchanged.
    result = optimize(config, initial_values, test_functions[0])
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.results, FunctionResults)
    assert result.results.variables is not None


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


def test_optimize_feeds_two_handlers_at_once(config: Any, test_functions: Any) -> None:
    first = HistoryHandler()
    second = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[first, second])
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


def test_optimize_local_handler_reused_by_concurrent_runs(
    config: Any, test_functions: Any
) -> None:
    single = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[single])

    history = HistoryHandler()
    results = optimize_many(
        [config, config],
        initial_values,
        test_functions[0],
        handlers=[history],
    )
    assert len(results) == 2
    assert len(history["results"]) > len(single["results"])


def test_optimize_local_handler_usable_after_error(config: Any) -> None:
    history = HistoryHandler()

    def _boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        optimize(config, initial_values, _boom, handlers=[history])
    # The failed run leaves nothing held, so the handler still takes events.
    assert history._event_owner is None  # ruff: ignore[private-member-access]


def test_optimize_handlers_do_not_nest_during_a_run(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    observed: list[int | None] = []

    def _report(_: FunctionResults) -> None:
        observed.append(history._event_owner)  # ruff: ignore[private-member-access]

    optimize(
        config,
        initial_values,
        test_functions[0],
        handlers=[history],
        report=_report,
    )
    # Handlers run one after the other, not nested, and each clears its owner
    # on the way out. Drop that reset and `history` reports itself as running
    # while the report handler is called.
    assert observed
    assert all(owner is None for owner in observed)


@pytest.mark.timeout(60)
def test_optimize_from_a_handler_reaching_itself_raises(
    config: Any, test_functions: Any
) -> None:
    class _NestingHandler(EventHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nested = False

        @property
        def event_types(self) -> set[EnOptEventType]:
            return {EnOptEventType.FINISHED_EVALUATION}

        def _handle_event(self, _event: EnOptEvent) -> None:
            if self.nested:
                return
            self.nested = True
            optimize(config, initial_values, test_functions[0], handlers=[self])

    handler = _NestingHandler()
    with pytest.raises(WorkflowError, match="already running further up this call"):
        optimize(config, initial_values, test_functions[0], handlers=[handler])
    assert handler.nested


@pytest.mark.timeout(60)
def test_optimize_from_a_handler_with_a_separate_handler_succeeds(
    config: Any, test_functions: Any
) -> None:
    inner = HistoryHandler()

    class _NestingHandler(EventHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nested = False

        @property
        def event_types(self) -> set[EnOptEventType]:
            return {EnOptEventType.FINISHED_EVALUATION}

        def _handle_event(self, _event: EnOptEvent) -> None:
            if self.nested:
                return
            self.nested = True
            optimize(config, initial_values, test_functions[0], handlers=[inner])

    handler = _NestingHandler()
    optimize(config, initial_values, test_functions[0], handlers=[handler])
    assert handler.nested
    assert inner["results"]


def test_report_callback_stops_optimization(config: Any, test_functions: Any) -> None:
    reported = 0

    def _report(_: FunctionResults) -> bool:
        nonlocal reported
        reported += 1
        return True

    result = optimize(config, initial_values, test_functions[0], report=_report)
    assert result.exit_code == ExitCode.USER_ABORT
    assert reported == 1


def test_report_callback_stops_only_own_run(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    def _stop(_: FunctionResults) -> bool:
        return True

    def _continue(_: FunctionResults) -> None:
        return None

    x0 = np.array([initial_values, initial_values])
    results = optimize_many(
        config,
        x0,
        test_functions[0],
        report=[_stop, _continue],
        executor=executors(workers=2),
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
    assert isinstance(result, FunctionResults)
    assert result.target_objective is not None
    assert result.target_objective == pytest.approx(0.66)
    assert result.functions is not None
    assert result.functions.objectives.shape == (1,)
    assert result.functions.constraints is None
    assert result.variables.shape == (initial_values.size,)


def test_evaluate_reports_the_evaluated_point(config: Any, test_functions: Any) -> None:
    result = evaluate(config, initial_values, test_functions[0])
    assert result.variables is not None
    assert np.array_equal(result.variables, initial_values)


def test_thread_run_sees_only_the_handlers_it_is_given(
    config: Any, test_functions: Any
) -> None:
    handler = HistoryHandler()

    def _without_handlers() -> None:
        optimize(config, initial_values, test_functions[0])

    thread = threading.Thread(target=_without_handlers)
    thread.start()
    thread.join()
    assert handler["results"] is None

    def _with_handlers() -> None:
        optimize(config, initial_values, test_functions[0], handlers=[handler])

    thread = threading.Thread(target=_with_handlers)
    thread.start()
    thread.join()
    assert handler["results"] is not None


def test_evaluate_feeds_one_handler_across_calls(
    config: Any, test_functions: Any
) -> None:
    handler = HistoryHandler()
    evaluate(config, initial_values, test_functions[0], handlers=[handler])
    evaluate(
        config, np.zeros(initial_values.size), test_functions[0], handlers=[handler]
    )
    assert len(handler["results"]) == 2


def test_evaluate_accepts_a_local_handler(config: Any, test_functions: Any) -> None:
    history = HistoryHandler()
    evaluate(config, initial_values, test_functions[0], handlers=[history])
    assert len(history["results"]) == 1


def test_evaluate_many_accepts_a_local_handler(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    evaluate_many(config, matrix, test_functions[0], handlers=[history])
    assert len(history["results"]) == 2


def test_evaluate_report_return_value_ignored(config: Any, test_functions: Any) -> None:
    reported: list[FunctionResults] = []

    def _stop(result: FunctionResults) -> bool:
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
    reported: list[FunctionResults] = []

    def _stop(result: FunctionResults) -> bool:
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
    assert all(isinstance(result, FunctionResults) for result in results)
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
    assert result.functions is not None
    assert result.functions.objectives.shape == (2,)
    assert result.functions.constraints is None


def test_evaluate_attaches_metadata_to_results(
    config: Any, test_functions: Any
) -> None:
    result = evaluate(
        config, initial_values, test_functions[0], metadata={"tag": "eval"}
    )
    assert result.metadata["tag"] == "eval"


def test_evaluate_many_attaches_metadata_to_every_result(
    config: Any, test_functions: Any
) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    results = evaluate_many(config, matrix, test_functions[0], metadata={"tag": "eval"})
    assert len(results) == 2
    for result in results:
        assert result.metadata["tag"] == "eval"


def test_optimize_with_a_thread_executor(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    result = optimize(
        config,
        initial_values,
        test_functions[0],
        executor=executors(workers=2),
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.5, atol=0.02)


def _collect_batch_ids(sink: list[int], lock: threading.Lock) -> Any:
    def _function(
        _variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        with lock:
            sink.append(context.batch_id)
        return 0.0

    return _function


def test_optimize_evaluates_on_the_given_executor(
    executors: Callable[..., Executor], config: Any
) -> None:
    # An executor is only ever what a run is handed; proving the evaluation
    # lands on a worker thread, not the caller's, is what distinguishes a run
    # that really dispatches from one that silently fell back to in-process.
    seen: list[str] = []
    lock = threading.Lock()

    def _record_thread(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        with lock:
            seen.append(threading.current_thread().name)
        return 0.0

    optimize(config, initial_values, _record_thread, executor=executors(workers=2))
    assert seen
    assert threading.current_thread().name not in seen


def test_optimize_without_an_executor_evaluates_in_process(config: Any) -> None:
    # The mirror of the above: with no executor passed, a run must evaluate on
    # the calling thread rather than reaching for one from its surroundings.
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


@pytest.mark.parametrize("arrangement", ["shared", "separate", "none"])
def test_batch_ids_are_unique_across_sequential_runs(
    executors: Callable[..., Executor], config: Any, arrangement: str
) -> None:
    # One program-wide counter, so no arrangement of executors can repeat an id.
    if arrangement == "shared":
        shared = executors(workers=2)
        pair: list[Executor | None] = [shared, shared]
    elif arrangement == "separate":
        pair = [executors(workers=2), executors(workers=2)]
    else:
        pair = [None, None]
    sinks: list[list[int]] = [[], []]
    lock = threading.Lock()
    for sink, executor in zip(sinks, pair, strict=True):
        optimize(
            config, initial_values, _collect_batch_ids(sink, lock), executor=executor
        )
    assert all(sinks)
    assert not set(sinks[0]) & set(sinks[1])


@pytest.mark.parametrize("with_executor", [True, False])
def test_batch_ids_are_unique_across_concurrent_runs(
    executors: Callable[..., Executor], config: Any, *, with_executor: bool
) -> None:
    sinks: list[list[int]] = [[], [], []]
    lock = threading.Lock()
    optimize_many(
        config,
        initial_values,
        [_collect_batch_ids(sink, lock) for sink in sinks],
        executor=executors(workers=2) if with_executor else None,
    )
    assert all(sinks)
    assert sum(len(set(sink)) for sink in sinks) == len(set().union(*sinks))


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


def test_metadata_reaches_the_evaluation_function_with_an_executor(
    executors: Callable[..., Executor], config: Any
) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    optimize(
        config,
        initial_values,
        _record_metadata(seen, lock),
        metadata={"run": 7},
        executor=executors(workers=2),
    )
    assert seen
    assert all(item == {"run": 7} for item in seen)


def test_metadata_reaches_the_evaluation_function_of_evaluate(config: Any) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    evaluate(config, initial_values, _record_metadata(seen, lock), metadata={"run": 7})
    assert seen
    assert all(item == {"run": 7} for item in seen)


def test_metadata_per_run_reaches_each_evaluation_function(
    executors: Callable[..., Executor], config: Any
) -> None:
    first: list[Any] = []
    second: list[Any] = []
    lock = threading.Lock()
    optimize_many(
        config,
        initial_values,
        [_record_metadata(first, lock), _record_metadata(second, lock)],
        metadata=[{"run": 0}, {"run": 1}],
        executor=executors(workers=2),
    )
    assert all(item == {"run": 0} for item in first)
    assert all(item == {"run": 1} for item in second)


def test_evaluate_many_with_a_thread_executor(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    results = evaluate_many(
        config, matrix, test_functions[0], executor=executors(workers=2)
    )
    for result, expected in zip(results, [0.66, 0.75], strict=True):
        assert result.target_objective == pytest.approx(expected)


def test_evaluate_with_a_thread_executor(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    result = evaluate(
        config,
        initial_values,
        test_functions[0],
        executor=executors(workers=2),
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
    # A run's evaluation function is plain code: it may build an executor of its
    # own, nested inside whatever executor is running it. Nothing ambient needs
    # to be threaded through for that to work.
    with ThreadExecutor(workers=1) as executor:
        result = optimize(_INNER_CONFIG, variables, _sphere, executor=executor)
    assert result.results is not None
    assert result.results.target_objective is not None
    return float(result.results.target_objective)


def _offload_from_evaluation(_variables: NDArray[np.float64], _context: Any) -> float:
    return float(offload(_return_one))


def _return_one() -> float:
    return 1.0


def test_evaluation_function_can_open_its_own_thread_executor(
    executors: Callable[..., Executor], config: Any
) -> None:
    result = optimize(
        config,
        initial_values,
        _run_inner_optimization,
        executor=executors(workers=2),
    )
    assert result.results is not None


@pytest.mark.slow
def test_evaluation_function_can_open_its_own_process_executor(
    executors: Callable[..., Executor], config: Any
) -> None:
    result = optimize(
        config,
        initial_values,
        _run_inner_optimization,
        executor=executors(ProcessExecutor, workers=2),
    )
    assert result.results is not None


_BUNDLE_CONFIG: dict[str, Any] = {
    "variables": {"variable_count": 2},
    "realizations": {"weights": [1.0] * 4},
}


def _bundle_pid(
    variables: NDArray[np.float64], _context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(
        objectives=float(np.sum(variables**2)), metadata={"pid": os.getpid()}
    )


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_bundle_size_sends_the_whole_batch_to_one_worker(
    executors: Callable[..., Executor],
) -> None:
    # The evaluations in one bundle run after each other, so a whole-batch
    # bundle is observable as a single worker doing all four realizations.
    history = HistoryHandler()
    executor = executors(ProcessExecutor, workers=4)
    evaluate(
        _BUNDLE_CONFIG,
        np.zeros(2),
        _bundle_pid,
        executor=executor,
        handlers=[history],
        bundle_size=0,
    )
    pids: set[int] = set()
    for item in history["results"]:
        recorded = item.evaluations.metadata.get("pid")
        if recorded is not None:
            pids.update(int(pid) for pid in np.ravel(recorded))
    assert len(pids) == 1


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_call_bundle_size_reaches_the_executor(
    executors: Callable[..., Executor], monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[int] = []
    executor = executors(ProcessExecutor, workers=4)
    submit = ProcessExecutor._submit  # ruff: ignore[private-member-access]

    def _recording_submit(self: ProcessExecutor, bundle: list[Any]) -> Any:
        sizes.append(len(bundle))
        return submit(self, bundle)

    monkeypatch.setattr(ProcessExecutor, "_submit", _recording_submit)
    evaluate(_BUNDLE_CONFIG, np.zeros(2), _bundle_pid, executor=executor, bundle_size=0)
    # Without the argument this would have been [1, 1, 1, 1].
    assert sizes == [4]


def test_negative_call_bundle_size_refused(executors: Callable[..., Executor]) -> None:
    with (
        pytest.raises(ValueError, match="bundle_size must be >= 0"),
    ):
        evaluate(
            _BUNDLE_CONFIG,
            np.zeros(2),
            _bundle_pid,
            executor=executors(),
            bundle_size=-1,
        )


def test_thread_executor_bundles_a_whole_batch_onto_one_thread(
    executors: Callable[..., Executor],
) -> None:
    # A thread executor used to ignore bundle_size. It no longer does: a whole
    # batch in one bundle is one worker task, so one thread runs all four.
    threads: set[int] = set()
    lock = threading.Lock()

    def _record_thread(
        variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        with lock:
            threads.add(threading.get_ident())
        return EvaluationFunctionResult(objectives=float(np.sum(variables**2)))

    results = evaluate(
        _BUNDLE_CONFIG,
        np.zeros(2),
        _record_thread,
        executor=executors(workers=4),
        bundle_size=0,
    )
    assert results.functions is not None
    assert len(threads) == 1


def test_thread_executor_runs_unbundled_calls_at_once(
    executors: Callable[..., Executor],
) -> None:
    # The counterpart: one call per bundle, so none of the four can pass the
    # barrier until all four are in flight. Bundling would break it instead.
    barrier = threading.Barrier(4)

    def _wait_for_all(
        variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        barrier.wait(timeout=30)
        return EvaluationFunctionResult(objectives=float(np.sum(variables**2)))

    results = evaluate(
        _BUNDLE_CONFIG,
        np.zeros(2),
        _wait_for_all,
        executor=executors(workers=4),
        bundle_size=1,
    )
    assert results.functions is not None


def test_bundle_size_sequence_length_must_match_runs() -> None:
    with pytest.raises(ValueError, match="bundle_size sequence length"):
        optimize_many(
            _BUNDLE_CONFIG,
            np.zeros(2),
            [_bundle_pid, _bundle_pid],
            bundle_size=[1, 2, 3],
        )


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
    executor: Executor,
    history: HistoryHandler,
    barrier: threading.Barrier,
) -> float:
    # Neither outer evaluation can pass until both have arrived, so if they were
    # run one after the other this breaks the barrier instead of quietly passing.
    barrier.wait()
    result = optimize(
        _NESTED_INNER,
        variables,
        _pid_sphere,
        executor=executor,
        handlers=[history],
        metadata={"outer": context.eval_idx},
    )
    assert result.results is not None
    assert result.results.target_objective is not None
    return float(result.results.target_objective)


@pytest.mark.parametrize(
    "processes",
    [
        pytest.param(False, id="threads"),
        pytest.param(True, id="processes", marks=pytest.mark.slow),
    ],
)
@pytest.mark.timeout(120)
def test_concurrent_inner_runs_on_a_second_executor_feed_one_handler(
    executors: Callable[..., Executor],
    processes: Any,
) -> None:
    history = HistoryHandler()
    barrier = threading.Barrier(len(_NESTED_POINTS), timeout=30)
    inner = executors(ProcessExecutor, workers=2) if processes else executors(workers=2)
    outer = executors(workers=len(_NESTED_POINTS))
    evaluate_many(
        _NESTED_OUTER,
        _NESTED_POINTS,
        partial(_nested_run, executor=inner, history=history, barrier=barrier),
        executor=outer,
    )

    batches: dict[int, set[int]] = {}
    pids: set[int] = set()
    for item in history["results"]:
        batches.setdefault(item.metadata["outer"], set()).add(item.batch_id)
        recorded = item.evaluations.metadata.get("pid")
        if recorded is not None:
            pids.update(int(pid) for pid in np.ravel(recorded))
    # Both inner runs reached the one handler, each tagged with the outer
    # evaluation that started it.
    assert set(batches) == {0, 1}
    # Batch ids come from one program-wide counter, so they never collided.
    assert not batches[0] & batches[1]
    assert pids
    if processes:
        # A process executor evaluates in workers of its own.
        assert os.getpid() not in pids
    else:
        # A thread executor evaluates here, so nesting needs no picklable function.
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
    with ThreadExecutor(workers=1) as executor:
        inner = optimize(
            _BILEVEL_CONFIG,
            [0.0],
            partial(_inner_objective, outer_value=a),
            executor=executor,
        )
    assert inner.results is not None
    assert inner.results.target_objective is not None
    return float(inner.results.target_objective)


def test_nested_optimization_on_a_thread_executor(
    executors: Callable[..., Executor],
) -> None:
    result = optimize(
        _BILEVEL_CONFIG, [0.0], _bilevel_outer, executor=executors(workers=1)
    )
    assert result.results is not None
    assert result.results.variables[0] == pytest.approx(2.0, abs=0.05)
    assert result.results.target_objective == pytest.approx(0.0, abs=1e-2)


def test_offload_in_evaluation_without_an_executor_runs_inline(
    executors: Callable[..., Executor], config: Any
) -> None:
    # An executor is only ever what is passed to a call, never what is found:
    # `offload` inside the evaluation function is given none, so it runs
    # inline even though the run itself is evaluating on a thread executor.
    result = optimize(
        config,
        initial_values,
        _offload_from_evaluation,
        executor=executors(workers=2),
    )
    assert result.results is not None


class _FatalWork(BaseException):
    """Not an Exception, so nothing on the way back is tempted to handle it."""


def _fatal_work() -> int:
    msg = "worker died"
    raise _FatalWork(msg)


def test_fatal_worker_error_reaches_the_caller(
    executors: Callable[..., Executor],
) -> None:
    # A worker cannot act on a BaseException, so it travels to the caller
    # unchanged rather than being folded into a group along the way.
    with pytest.raises(_FatalWork, match="worker died"):
        offload(_fatal_work, executor=executors(workers=1))


def _double(value: float) -> float:
    return 2.0 * value


def _offload_in_own_executor(variables: NDArray[np.float64], _context: Any) -> float:
    with ThreadExecutor(workers=2) as executor:
        doubled = offload(
            [partial(_double, 3.0), partial(_double, 4.0)], executor=executor
        )
    assert doubled == (6.0, 8.0)
    return float(variables @ variables)


def _own_handlers_in_evaluation(variables: NDArray[np.float64], _context: Any) -> float:
    # A run's evaluation function is plain code: it may start a run with
    # handlers of its own, nested inside whatever executor is running it.
    history = HistoryHandler()
    optimize(_INNER_CONFIG, variables, _sphere, handlers=[history])
    assert len(history.results) > 0
    return float(variables @ variables)


def test_offload_in_evaluation_uses_its_own_executor(
    executors: Callable[..., Executor], config: Any
) -> None:
    result = optimize(
        config,
        initial_values,
        _offload_in_own_executor,
        executor=executors(workers=2),
    )
    assert result.results is not None


def test_handlers_in_an_evaluation(
    executors: Callable[..., Executor], config: Any
) -> None:
    result = optimize(
        config,
        initial_values,
        _own_handlers_in_evaluation,
        executor=executors(workers=2),
    )
    assert result.results is not None


@pytest.mark.slow
def test_sequential_executors_are_allowed(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    first = optimize(
        config,
        initial_values,
        test_functions[0],
        executor=executors(workers=2),
    )
    second = optimize(
        config,
        initial_values,
        test_functions[0],
        executor=executors(ProcessExecutor, workers=2),
    )
    assert first.exit_code == second.exit_code
    assert first.results is not None
    assert second.results is not None
    assert first.results.variables == pytest.approx(second.results.variables)


def test_thread_executor_objective_exception_propagates(
    executors: Callable[..., Executor], config: Any
) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        optimize(config, initial_values, boom, executor=executors(workers=2))


def test_thread_executor_survives_objective_exception(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    executor = executors(workers=2)
    with pytest.raises(ValueError, match="boom"):
        optimize(config, initial_values, boom, executor=executor)
    # The executor survives a failed run and can still be used by the next one.
    result = optimize(config, initial_values, test_functions[0], executor=executor)
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_optimize_with_a_process_executor(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    result = optimize(
        config,
        initial_values,
        test_functions[0],
        executor=executors(ProcessExecutor, workers=2),
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_process_executor_without_cloudpickle(
    executors: Callable[..., Executor], config: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._process_executor.dumps",
        pickle.dumps,
    )
    result = optimize(
        config, initial_values, _sphere, executor=executors(ProcessExecutor, workers=2)
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.0, atol=0.02)


@pytest.mark.slow
def test_evaluate_without_cloudpickle(
    executors: Callable[..., Executor], config: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._process_executor.dumps",
        pickle.dumps,
    )
    result = evaluate(
        config, initial_values, _sphere, executor=executors(ProcessExecutor, workers=2)
    )
    assert result.target_objective == pytest.approx(0.01)


def test_optimize_many_broadcasts_config_and_objective(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = optimize_many(
        config, starts, test_functions[0], executor=executors(workers=2)
    )
    assert len(results) == 2
    assert all(isinstance(result, OptimizationResult) for result in results)
    for result in results:
        assert result.results is not None
        assert np.allclose(result.results.variables, 0.5, atol=0.02)


def test_optimize_many_per_run_objectives(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    results = optimize_many(
        config,
        initial_values,
        [test_functions[0], test_functions[1]],
        executor=executors(workers=2),
    )
    assert len(results) == 2
    assert results[0].results is not None
    assert results[1].results is not None
    assert np.allclose(results[0].results.variables, [0.5, 0.5, 0.5], atol=0.02)
    assert np.allclose(results[1].results.variables, [-1.5, -1.5, 0.5], atol=0.02)


def test_optimize_many_report_callback_shared_across_runs(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    reported: list[FunctionResults] = []
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    optimize_many(
        config,
        starts,
        test_functions[0],
        report=reported.append,
        executor=executors(workers=2),
    )
    assert reported
    assert all(isinstance(item, FunctionResults) for item in reported)


def test_optimize_many_accepts_a_report_per_run(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    first: list[FunctionResults] = []
    second: list[FunctionResults] = []
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    optimize_many(
        config,
        starts,
        test_functions[0],
        report=[first.append, second.append],
        executor=executors(workers=2),
    )
    assert first
    assert second
    assert all(isinstance(item, FunctionResults) for item in (*first, *second))


def test_optimize_many_rejects_mismatched_report_sequence(
    config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    with pytest.raises(ValueError, match="number of runs"):
        optimize_many(config, starts, test_functions[0], report=[lambda _r: None])


def test_optimize_many_broadcasts_a_single_metadata_dict(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = optimize_many(
        config,
        starts,
        test_functions[0],
        metadata={"group": "g"},
        executor=executors(workers=2),
    )
    for result in results:
        assert result.results is not None
        assert result.results.metadata["group"] == "g"


def test_optimize_many_accepts_metadata_per_run(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = optimize_many(
        config,
        starts,
        test_functions[0],
        metadata=[{"run_id": 0}, {"run_id": 1}],
        executor=executors(workers=2),
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


def test_optimize_many_respects_limit(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    starts = np.tile(initial_values, (4, 1))
    results = optimize_many(
        config,
        starts,
        test_functions[0],
        limit=2,
        executor=executors(workers=2),
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


def test_optimize_many_without_an_executor(config: Any, test_functions: Any) -> None:
    # Without an executor, optimize_many evaluates on its own driver threads.
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = optimize_many(config, starts, test_functions[0])
    assert len(results) == 2
    for result in results:
        assert result.results is not None
        assert np.allclose(result.results.variables, 0.5, atol=0.02)


def test_optimize_many_fail_fast(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    starts = np.tile(initial_values, (3, 1))
    with pytest.raises(ValueError, match="boom"):
        optimize_many(
            config,
            starts,
            [test_functions[0], boom, test_functions[0]],
            executor=executors(workers=2),
        )


@pytest.mark.timeout(60)
def test_optimize_many_skips_runs_that_have_not_started(
    executors: Callable[..., Executor], config: Any
) -> None:
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
    # only costs the ceiling when no run does. The executor outlives the wait,
    # so a pending run is refused by the stop flag, not by a closed executor.
    starts = np.tile(initial_values, (5, 1))
    executor = executors(workers=2)
    with pytest.raises(ValueError, match="boom"):
        optimize_many(config, starts, boom, limit=1, executor=executor)
    assert not ran_again.wait(timeout=0.2)
    with lock:
        assert calls == 1


@pytest.mark.timeout(60)
def test_optimize_many_leaves_the_executor_usable_after_a_failure(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    # Every run reaches its first evaluation before any of them returns, so the
    # three siblings are provably in flight when the fourth fails and are
    # abandoned rather than skipped. A barrier gives that ordering outright;
    # sleeping for it only makes it likely. One worker per run, so the
    # rendezvous cannot starve on the executor.
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
    executor = executors(workers=4)
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
            executor=executor,
        )
    # Siblings are abandoned, not cancelled, so they may still be running
    # here; the executor must stay usable and must not deadlock against them.
    assert not executor.closed
    result = optimize(config, initial_values, test_functions[0], executor=executor)
    assert result.exit_code == ExitCode.OPTIMIZER_FINISHED


@pytest.mark.timeout(60)
def test_run_abandoned_by_fail_fast_returns(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    # `run_concurrent` is the primitive `optimize_many` runs its drivers on. A
    # run already in flight when a sibling fails cannot be cancelled, and
    # nothing takes its executor away, so it runs to completion and *returns* a
    # result. That is why a fail-fast failure never sprays exceptions out of
    # its driver threads. (A close under an abandoned run is a different case;
    # `test_guards.py` covers it.)
    outcomes: list[Any] = []
    lock = threading.Lock()
    started = threading.Barrier(4)
    finished = threading.Semaphore(0)

    def first_evaluation_waits() -> Any:
        # The rendezvous sits inside the run, not on the driver thread before
        # it: reaching it proves the run is past the entry check and is really
        # in flight when the sibling below fails.
        waited = False

        def objective(variables: Any, context: Any) -> float:
            nonlocal waited
            if not waited:
                waited = True
                started.wait(timeout=30)
            return float(test_functions[0](variables, context))

        return objective

    def record(executor: Executor) -> None:
        try:
            result = optimize(
                config, initial_values, first_evaluation_waits(), executor=executor
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
        # run. One worker per run, so the rendezvous cannot starve on it.
        started.wait(timeout=30)
        msg = "boom"
        raise ValueError(msg)

    executor = executors(workers=4)
    jobs: list[Callable[[], None]] = [
        partial(record, executor),
        partial(record, executor),
        boom,
        partial(record, executor),
    ]
    with pytest.raises(ValueError, match="boom"):
        run_concurrent(jobs)

    # Each abandoned run releases the semaphore as it ends, so this returns as
    # soon as the last one does; the timeout is only a ceiling on a hang.
    for _ in range(3):
        assert finished.acquire(timeout=30), "abandoned runs never finished"

    assert outcomes == [ExitCode.OPTIMIZER_FINISHED] * 3


def test_shared_handler_without_an_executor_aggregates_runs(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[history])
    optimize(config, initial_values, test_functions[0], handlers=[history])
    assert history["results"]


def test_shared_handler_aggregates_across_optimize_many(
    executors: Callable[..., Executor], config: Any, test_functions: Any
) -> None:
    single = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[single])

    shared = HistoryHandler()
    starts = np.tile(initial_values, (3, 1))
    optimize_many(
        config,
        starts,
        test_functions[0],
        executor=executors(workers=2),
        handlers=[shared],
    )

    assert len(shared["results"]) > len(single["results"])


def test_optimize_many_accepts_bare_handler(config: Any, test_functions: Any) -> None:
    single = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[single])

    handler = HistoryHandler()
    results = optimize_many(
        [config, config, config],
        initial_values,
        test_functions[0],
        handlers=[handler],
    )
    assert len(results) == 3
    # Every run feeds the same handler, which serializes the calls itself.
    assert len(handler["results"]) > len(single["results"])


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

        def live_job_ids(self) -> set[int]:
            running = [
                job_id
                for job_id, job_name in self._jobs.items()
                if not (self._path / f"{job_name}.out").exists()
            ]
            self._jobs = {job_id: self._jobs[job_id] for job_id in running}
            return set(self._jobs)

    def _mock_scheduler(monkeypatch: Any, adapter: _MockedHPCAdapter) -> None:
        # `ropt` asks the scheduler for the ids of the jobs that are still
        # there; that the real one answers with a table is `pysqa`'s business,
        # so the mocks never build one.
        monkeypatch.setattr(
            "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
            lambda *args, **kwargs: adapter,  # ruff: ignore[unused-lambda-argument]
        )
        monkeypatch.setattr(
            HPCExecutor, "_live_job_ids", lambda _self: adapter.live_job_ids()
        )


@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_evaluates_through_the_simple_api(
    executors: Callable[..., Executor],
    config: Any,
    test_functions: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _mock_scheduler(monkeypatch, _MockedHPCAdapter(tmp_path))
    result = evaluate(
        config,
        initial_values,
        test_functions[0],
        executor=executors(HPCExecutor, workers=2, workdir=tmp_path, template=""),
    )
    assert result.target_objective is not None


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_local_jobs_evaluate_through_the_simple_api(
    executors: Callable[..., Executor], config: Any, test_functions: Any, tmp_path: Path
) -> None:
    result = evaluate(
        config,
        initial_values,
        test_functions[0],
        executor=executors(LocalJobExecutor, workers=2, workdir=tmp_path),
    )
    assert result.target_objective is not None


_DYING_WORKER_CONFIG: dict[str, Any] = {
    "variables": {"variable_count": 2},
    "realizations": {"weights": [1.0] * 4, "realization_min_success": 1},
}


def _kill_worker_on_one_realization(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    if context.realization == 1:
        os._exit(1)
    return float(np.sum(variables**2))


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_dying_worker_raises_instead_of_failing_a_realization(
    executors: Callable[..., Executor],
) -> None:
    # A minimum of one success is enough to absorb the loss, so without the
    # raise this returns an answer computed from the workers that survived.
    with pytest.raises(ExecutionError, match="could not be run"):
        evaluate(
            _DYING_WORKER_CONFIG,
            np.zeros(2),
            _kill_worker_on_one_realization,
            executor=executors(ProcessExecutor, workers=2),
        )
