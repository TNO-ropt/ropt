"""Tests for the sequential high-level ``optimize`` API."""

# The monkeypatched tests here name their target as an attribute and assert it
# was used. An earlier version patched a method by string and became a no-op the
# day it was renamed, which mypy cannot see. The other traps in this file are
# explained where they sit.

from __future__ import annotations

import os
import pickle  # ruff: ignore[suspicious-pickle-import]
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.event_handlers import EventHandler
from ropt.components.executors import (
    HPCExecutor,
    LocalJobExecutor,
    ProcessExecutor,
    ThreadExecutor,
)
from ropt.enums import EnOptEventType, ExitReason
from ropt.exceptions import ExecutionError, RunsFailedError, WorkflowError
from ropt.results import FunctionResults
from ropt.simple import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    HistoryHandler,
    OptimizationResult,
    evaluate,
    evaluate_batch,
    optimize,
    optimize_many,
    session,
)
from ropt.simple._function import adapt_function

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from ropt.components.executors import Executor
    from ropt.events import EnOptEvent
    from ropt.simple import WorkerPool

try:
    # The job path needs no extras of its own, so these tests run either way.
    import pysqa  # ruff: ignore[unused-import]

    from ropt.components.executors.__main__ import run_task

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False

initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="pools")
def pools_fixture() -> Iterator[Callable[..., WorkerPool]]:
    """Build pools on one session.

    Yields:
        A factory taking an executor class (`ThreadExecutor` by default) and its
        keyword arguments.
    """
    with session() as opened:
        factories: dict[type[Executor], Callable[..., WorkerPool]] = {
            ThreadExecutor: opened.thread_pool,
            ProcessExecutor: opened.process_pool,
            LocalJobExecutor: opened.local_pool,
            HPCExecutor: opened.hpc_pool,
        }

        def _make(kind: type[Executor] = ThreadExecutor, **kwargs: Any) -> WorkerPool:
            return factories[kind](**kwargs)

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
    assert result.exit_reason == ExitReason.FINISHED
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
    assert result.exit_reason == ExitReason.TOO_FEW_REALIZATIONS
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
    assert result.exit_reason == ExitReason.STOPPED
    assert reported == 1


def test_report_callback_stops_only_own_run(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    def _stop(_: FunctionResults) -> bool:
        return True

    def _continue(_: FunctionResults) -> None:
        return None

    x0 = np.array([initial_values, initial_values])
    results = pools(workers=2).optimize_many(
        config,
        x0,
        test_functions[0],
        report=[_stop, _continue],
    )
    assert results[0].exit_reason == ExitReason.STOPPED
    assert results[1].exit_reason != ExitReason.STOPPED


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
    outcome = evaluate(config, initial_values, test_functions[0])
    assert outcome.exit_reason is ExitReason.FINISHED
    result = outcome.results
    assert isinstance(result, FunctionResults)
    assert result.target_objective is not None
    assert result.target_objective == pytest.approx(0.66)
    assert result.functions is not None
    assert result.functions.objectives.shape == (1,)
    assert result.functions.constraints is None
    assert result.variables.shape == (initial_values.size,)


def test_evaluate_reports_the_evaluated_point(config: Any, test_functions: Any) -> None:
    result = evaluate(config, initial_values, test_functions[0]).results
    assert result is not None
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


def test_evaluate_batch_accepts_a_local_handler(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    evaluate_batch(config, matrix, test_functions[0], handlers=[history])
    assert len(history["results"]) == 2


def test_evaluate_report_return_value_ignored(config: Any, test_functions: Any) -> None:
    reported: list[FunctionResults] = []

    def _stop(result: FunctionResults) -> bool:
        reported.append(result)
        return True

    result = evaluate(config, initial_values, test_functions[0], report=_stop).results
    assert len(reported) == 1
    assert result is not None
    assert result.target_objective == pytest.approx(0.66)


def test_evaluate_batch_report_return_value_ignored(
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
    results = evaluate_batch(config, matrix, test_functions[0], report=_stop).results
    assert len(reported) == 1
    assert len(results) == 2


def test_evaluate_rejects_matrix(config: Any, test_functions: Any) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    with pytest.raises(ValueError, match="single vector"):
        evaluate(config, matrix, test_functions[0])


def test_evaluate_batch_returns_result_per_row(
    config: Any, test_functions: Any
) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    results = evaluate_batch(config, matrix, test_functions[0]).results
    assert len(results) == 2
    assert all(isinstance(result, FunctionResults) for result in results)
    # Squared distance to [0.5, 0.5, 0.5]: row 0 = 0.5^2+0.5^2+0.4^2, row 1 = 3*0.5^2.
    for result, expected in zip(results, [0.66, 0.75], strict=True):
        assert result.target_objective == pytest.approx(expected)


def test_evaluate_batch_single_row(config: Any, test_functions: Any) -> None:
    results = evaluate_batch(
        config, initial_values.reshape(1, -1), test_functions[0]
    ).results
    assert len(results) == 1
    assert results[0].target_objective == pytest.approx(0.66)


def test_evaluate_batch_rejects_vector(config: Any, test_functions: Any) -> None:
    with pytest.raises(ValueError, match="2-D matrix"):
        evaluate_batch(config, initial_values, test_functions[0])


def test_evaluate_multiple_objectives(config: Any, eval_func: Any) -> None:
    config["objectives"] = {"weights": [0.75, 0.25]}
    result = evaluate(config, initial_values, eval_func()).results
    assert result is not None
    assert result.functions is not None
    assert result.functions.objectives.shape == (2,)
    assert result.functions.constraints is None


def test_evaluate_attaches_metadata_to_results(
    config: Any, test_functions: Any
) -> None:
    result = evaluate(
        config, initial_values, test_functions[0], metadata={"tag": "eval"}
    ).results
    assert result is not None
    assert result.metadata["tag"] == "eval"


def test_evaluate_batch_attaches_metadata_to_every_result(
    config: Any, test_functions: Any
) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    results = evaluate_batch(
        config, matrix, test_functions[0], metadata={"tag": "eval"}
    ).results
    assert len(results) == 2
    for result in results:
        assert result.metadata["tag"] == "eval"


def test_optimize_with_a_thread_pool(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    result = pools(workers=2).optimize(config, initial_values, test_functions[0])
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


def test_optimize_evaluates_on_the_given_pool(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    # A pool is only ever what a run is handed; proving the evaluation
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

    pools(workers=2).optimize(config, initial_values, _record_thread)
    assert seen
    assert threading.current_thread().name not in seen


def test_optimize_without_a_pool_evaluates_in_process(config: Any) -> None:
    # The mirror of the above: with no pool passed, a run must evaluate on
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
    pools: Callable[..., WorkerPool], config: Any, arrangement: str
) -> None:
    # One program-wide counter, so no arrangement of pools can repeat an id.
    if arrangement == "shared":
        shared = pools(workers=2)
        pair: list[Callable[..., OptimizationResult]] = [shared.optimize] * 2
    elif arrangement == "separate":
        pair = [pools(workers=2).optimize, pools(workers=2).optimize]
    else:
        pair = [optimize, optimize]
    sinks: list[list[int]] = [[], []]
    lock = threading.Lock()
    for sink, run in zip(sinks, pair, strict=True):
        run(config, initial_values, _collect_batch_ids(sink, lock))
    assert all(sinks)
    assert not set(sinks[0]) & set(sinks[1])


@pytest.mark.parametrize("with_pool", [True, False])
def test_batch_ids_are_unique_across_concurrent_runs(
    pools: Callable[..., WorkerPool], config: Any, *, with_pool: bool
) -> None:
    sinks: list[list[int]] = [[], [], []]
    lock = threading.Lock()
    run = pools(workers=2).optimize_many if with_pool else optimize_many
    run(
        config,
        initial_values,
        [_collect_batch_ids(sink, lock) for sink in sinks],
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


def test_metadata_reaches_the_evaluation_function_with_a_pool(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    seen: list[Any] = []
    lock = threading.Lock()
    pools(workers=2).optimize(
        config,
        initial_values,
        _record_metadata(seen, lock),
        metadata={"run": 7},
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
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    first: list[Any] = []
    second: list[Any] = []
    lock = threading.Lock()
    pools(workers=2).optimize_many(
        config,
        initial_values,
        [_record_metadata(first, lock), _record_metadata(second, lock)],
        metadata=[{"run": 0}, {"run": 1}],
    )
    assert all(item == {"run": 0} for item in first)
    assert all(item == {"run": 1} for item in second)


def test_evaluate_batch_with_a_thread_pool(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    matrix = np.array([initial_values, np.zeros(initial_values.size)])
    outcome = pools(workers=2).evaluate_batch(config, matrix, test_functions[0])
    for result, expected in zip(outcome.results, [0.66, 0.75], strict=True):
        assert result.target_objective == pytest.approx(expected)


def test_evaluate_with_a_thread_pool(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    outcome = pools(workers=2).evaluate(config, initial_values, test_functions[0])
    assert outcome.results is not None
    assert outcome.results.target_objective == pytest.approx(0.66)


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
    with session() as inner:
        result = inner.thread_pool(workers=1).optimize(
            _INNER_CONFIG, variables, _sphere
        )
    assert result.results is not None
    assert result.results.target_objective is not None
    return float(result.results.target_objective)


def test_evaluation_function_can_open_its_own_thread_pool(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    result = pools(workers=2).optimize(config, initial_values, _run_inner_optimization)
    assert result.results is not None


@pytest.mark.slow
def test_evaluation_function_can_open_its_own_process_pool(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    result = pools(ProcessExecutor, workers=2).optimize(
        config, initial_values, _run_inner_optimization
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
    pools: Callable[..., WorkerPool],
) -> None:
    # The evaluations in one bundle run after each other, so a whole-batch
    # bundle is observable as a single worker doing all four realizations.
    history = HistoryHandler()
    pools(ProcessExecutor, workers=4).evaluate(
        _BUNDLE_CONFIG,
        np.zeros(2),
        _bundle_pid,
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
    pools: Callable[..., WorkerPool], monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[int] = []
    pool = pools(ProcessExecutor, workers=4)
    submit = ProcessExecutor._submit  # ruff: ignore[private-member-access]

    def _recording_submit(self: ProcessExecutor, bundle: list[Any]) -> Any:
        sizes.append(len(bundle))
        return submit(self, bundle)

    monkeypatch.setattr(ProcessExecutor, "_submit", _recording_submit)
    pool.evaluate(_BUNDLE_CONFIG, np.zeros(2), _bundle_pid, bundle_size=0)
    # Without the argument this would have been [1, 1, 1, 1].
    assert sizes == [4]


def test_negative_call_bundle_size_refused(pools: Callable[..., WorkerPool]) -> None:
    with (
        pytest.raises(ValueError, match="bundle_size must be >= 0"),
    ):
        pools().evaluate(
            _BUNDLE_CONFIG,
            np.zeros(2),
            _bundle_pid,
            bundle_size=-1,
        )


def test_thread_pool_bundles_a_whole_batch_onto_one_thread(
    pools: Callable[..., WorkerPool],
) -> None:
    # A thread pool used to ignore bundle_size. It no longer does: a whole
    # batch in one bundle is one worker task, so one thread runs all four.
    threads: set[int] = set()
    lock = threading.Lock()

    def _record_thread(
        variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        with lock:
            threads.add(threading.get_ident())
        return EvaluationFunctionResult(objectives=float(np.sum(variables**2)))

    results = pools(workers=4).evaluate(
        _BUNDLE_CONFIG,
        np.zeros(2),
        _record_thread,
        bundle_size=0,
    )
    assert results.results is not None
    assert results.results.functions is not None
    assert len(threads) == 1


def test_thread_pool_runs_unbundled_calls_at_once(
    pools: Callable[..., WorkerPool],
) -> None:
    # The counterpart: one call per bundle, so none of the four can pass the
    # barrier until all four are in flight. Bundling would break it instead.
    barrier = threading.Barrier(4)

    def _wait_for_all(
        variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        barrier.wait(timeout=30)
        return EvaluationFunctionResult(objectives=float(np.sum(variables**2)))

    results = pools(workers=4).evaluate(
        _BUNDLE_CONFIG,
        np.zeros(2),
        _wait_for_all,
        bundle_size=1,
    )
    assert results.results is not None
    assert results.results.functions is not None


def test_bundle_size_sequence_length_must_match_runs(
    pools: Callable[..., WorkerPool],
) -> None:
    with pytest.raises(ValueError, match="bundle_size sequence length"):
        pools().optimize_many(
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
    pool: WorkerPool,
    history: HistoryHandler,
    barrier: threading.Barrier,
) -> float:
    # Neither outer evaluation can pass until both have arrived, so if they were
    # run one after the other this breaks the barrier instead of quietly passing.
    barrier.wait()
    result = pool.optimize(
        _NESTED_INNER,
        variables,
        _pid_sphere,
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
def test_concurrent_inner_runs_on_a_second_pool_feed_one_handler(
    pools: Callable[..., WorkerPool],
    processes: Any,
) -> None:
    history = HistoryHandler()
    barrier = threading.Barrier(len(_NESTED_POINTS), timeout=30)
    inner = pools(ProcessExecutor, workers=2) if processes else pools(workers=2)
    outer = pools(workers=len(_NESTED_POINTS))
    outer.evaluate_batch(
        _NESTED_OUTER,
        _NESTED_POINTS,
        partial(_nested_run, pool=inner, history=history, barrier=barrier),
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
    with session() as inner_session:
        inner = inner_session.thread_pool(workers=1).optimize(
            _BILEVEL_CONFIG,
            [0.0],
            partial(_inner_objective, outer_value=a),
        )
    assert inner.results is not None
    assert inner.results.target_objective is not None
    return float(inner.results.target_objective)


def test_nested_optimization_on_a_thread_pool(
    pools: Callable[..., WorkerPool],
) -> None:
    result = pools(workers=1).optimize(_BILEVEL_CONFIG, [0.0], _bilevel_outer)
    assert result.results is not None
    assert result.results.variables[0] == pytest.approx(2.0, abs=0.05)
    assert result.results.target_objective == pytest.approx(0.0, abs=1e-2)


class _FatalWork(BaseException):
    """Not an Exception, so nothing on the way back is tempted to handle it."""


def _fatal_work() -> int:
    msg = "worker died"
    raise _FatalWork(msg)


def test_fatal_worker_error_reaches_the_caller(
    pools: Callable[..., WorkerPool],
) -> None:
    # A worker cannot act on a BaseException, so it travels to the caller
    # unchanged rather than being folded into a group along the way.
    with pytest.raises(_FatalWork, match="worker died"):
        pools(workers=1).offload(_fatal_work)


def _double(value: float) -> float:
    return 2.0 * value


def _offload_in_own_pool(variables: NDArray[np.float64], _context: Any) -> float:
    with session() as inner:
        doubled = inner.thread_pool(workers=2).offload(
            [partial(_double, 3.0), partial(_double, 4.0)],
        )
    assert doubled == (6.0, 8.0)
    return float(variables @ variables)


def _own_handlers_in_evaluation(variables: NDArray[np.float64], _context: Any) -> float:
    # A run's evaluation function is plain code: it may start a run with
    # handlers of its own, nested inside whatever pool is running it.
    history = HistoryHandler()
    optimize(_INNER_CONFIG, variables, _sphere, handlers=[history])
    assert len(history.results) > 0
    return float(variables @ variables)


def test_offload_in_evaluation_uses_its_own_pool(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    result = pools(workers=2).optimize(config, initial_values, _offload_in_own_pool)
    assert result.results is not None


def test_handlers_in_an_evaluation(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    result = pools(workers=2).optimize(
        config, initial_values, _own_handlers_in_evaluation
    )
    assert result.results is not None


@pytest.mark.slow
def test_sequential_pools_are_allowed(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    first = pools(workers=2).optimize(config, initial_values, test_functions[0])
    second = pools(ProcessExecutor, workers=2).optimize(
        config, initial_values, test_functions[0]
    )
    assert first.exit_reason == second.exit_reason
    assert first.results is not None
    assert second.results is not None
    assert first.results.variables == pytest.approx(second.results.variables)


def test_thread_pool_objective_exception_propagates(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        pools(workers=2).optimize(config, initial_values, boom)


def test_thread_pool_survives_objective_exception(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    pool = pools(workers=2)
    with pytest.raises(ValueError, match="boom"):
        pool.optimize(config, initial_values, boom)
    # The pool survives a failed run and can still be used by the next one.
    result = pool.optimize(config, initial_values, test_functions[0])
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_optimize_with_a_process_pool(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    result = pools(ProcessExecutor, workers=2).optimize(
        config, initial_values, test_functions[0]
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.5, atol=0.02)


@pytest.mark.slow
def test_process_pool_without_cloudpickle(
    pools: Callable[..., WorkerPool], config: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._process_executor.dumps",
        pickle.dumps,
    )
    result = pools(ProcessExecutor, workers=2).optimize(config, initial_values, _sphere)
    assert result.results is not None
    assert np.allclose(result.results.variables, 0.0, atol=0.02)


@pytest.mark.slow
def test_evaluate_without_cloudpickle(
    pools: Callable[..., WorkerPool], config: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._process_executor.dumps",
        pickle.dumps,
    )
    result = pools(ProcessExecutor, workers=2).evaluate(config, initial_values, _sphere)
    assert result.results is not None
    assert result.results.target_objective == pytest.approx(0.01)


def test_optimize_many_broadcasts_config_and_objective(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = pools(workers=2).optimize_many(config, starts, test_functions[0])
    assert len(results) == 2
    assert all(isinstance(result, OptimizationResult) for result in results)
    for result in results:
        assert result.results is not None
        assert np.allclose(result.results.variables, 0.5, atol=0.02)


def test_optimize_many_per_run_objectives(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    results = pools(workers=2).optimize_many(
        config,
        initial_values,
        [test_functions[0], test_functions[1]],
    )
    assert len(results) == 2
    assert results[0].results is not None
    assert results[1].results is not None
    assert np.allclose(results[0].results.variables, [0.5, 0.5, 0.5], atol=0.02)
    assert np.allclose(results[1].results.variables, [-1.5, -1.5, 0.5], atol=0.02)


def test_optimize_many_report_callback_shared_across_runs(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    reported: list[FunctionResults] = []
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    pools(workers=2).optimize_many(
        config,
        starts,
        test_functions[0],
        report=reported.append,
    )
    assert reported
    assert all(isinstance(item, FunctionResults) for item in reported)


def test_optimize_many_accepts_a_report_per_run(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    first: list[FunctionResults] = []
    second: list[FunctionResults] = []
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    pools(workers=2).optimize_many(
        config,
        starts,
        test_functions[0],
        report=[first.append, second.append],
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
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = pools(workers=2).optimize_many(
        config,
        starts,
        test_functions[0],
        metadata={"group": "g"},
    )
    for result in results:
        assert result.results is not None
        assert result.results.metadata["group"] == "g"


def test_optimize_many_accepts_metadata_per_run(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = pools(workers=2).optimize_many(
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
    with pytest.raises(ValueError, match="number of runs"):
        optimize_many(config, starts, test_functions[0], metadata=[{"run_id": 0}])


def test_optimize_many_respects_limit(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    starts = np.tile(initial_values, (4, 1))
    results = pools(workers=2).optimize_many(
        config,
        starts,
        test_functions[0],
        limit=2,
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


def test_optimize_many_without_a_pool(config: Any, test_functions: Any) -> None:
    # Without a pool, optimize_many evaluates on its own driver threads.
    starts = np.array([initial_values, np.zeros(initial_values.size)])
    results = optimize_many(config, starts, test_functions[0])
    assert len(results) == 2
    for result in results:
        assert result.results is not None
        assert np.allclose(result.results.variables, 0.5, atol=0.02)


def test_optimize_many_carries_the_outcome_of_every_run_when_one_fails(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    # The runs that did not fail were cut off by the one that did, so what they
    # reached is only reachable through the carrier.
    def boom(_v: Any, _c: Any) -> float:
        msg = "boom"
        raise ValueError(msg)

    starts = np.tile(initial_values, (3, 1))
    with pytest.raises(RunsFailedError) as raised:
        pools(workers=2).optimize_many(
            config,
            starts,
            [test_functions[0], boom, test_functions[0]],
        )

    outcomes = raised.value.outcomes
    assert len(outcomes) == 3
    assert isinstance(outcomes[1], ValueError)
    assert str(outcomes[1]) == "boom"
    assert isinstance(raised.value.__cause__, ValueError)
    for index in (0, 2):
        outcome = outcomes[index]
        assert isinstance(outcome, OptimizationResult)
        assert outcome.exit_reason in {
            ExitReason.ABORTED_ON_ERROR,
            ExitReason.FINISHED,
        }


@pytest.mark.timeout(60)
def test_optimize_many_cuts_off_queued_runs_when_one_fails(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    calls = 0
    lock = threading.Lock()

    def boom(_v: Any, _c: Any) -> float:
        nonlocal calls
        with lock:
            calls += 1
        msg = "boom"
        raise ValueError(msg)

    # One at a time, so the first run fails before any other is admitted.
    starts = np.tile(initial_values, (5, 1))
    pool = pools(workers=2)
    with pytest.raises(RunsFailedError) as raised:
        pool.optimize_many(config, starts, boom, limit=1)

    outcomes = raised.value.outcomes
    assert isinstance(outcomes[0], ValueError)
    for outcome in outcomes[1:]:
        assert isinstance(outcome, OptimizationResult)
        assert outcome.exit_reason == ExitReason.ABORTED_ON_ERROR
        assert outcome.results is None
    with lock:
        assert calls == 1


@pytest.mark.timeout(60)
def test_optimize_many_with_keep_going_starts_queued_runs_when_one_fails(
    pools: Callable[..., WorkerPool], config: Any
) -> None:
    calls = 0
    lock = threading.Lock()

    def boom(_v: Any, _c: Any) -> float:
        nonlocal calls
        with lock:
            calls += 1
        msg = "boom"
        raise ValueError(msg)

    starts = np.tile(initial_values, (5, 1))
    pool = pools(workers=2)
    with pytest.raises(RunsFailedError):
        pool.optimize_many(config, starts, boom, limit=1, keep_going=True)
    with lock:
        assert calls == 5


@pytest.mark.timeout(60)
def test_optimize_many_leaves_the_pool_usable_after_a_failure(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    # Every run reaches its first evaluation before any of them returns, so the
    # three siblings are provably in flight when the fourth fails. A barrier
    # gives that ordering outright; sleeping for it only makes it likely. One
    # worker per run, so the rendezvous cannot starve on the pool.
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
    pool = pools(workers=4)
    with pytest.raises(RunsFailedError):
        pool.optimize_many(
            config,
            starts,
            [
                first_evaluation_waits(),
                first_evaluation_waits(),
                boom,
                first_evaluation_waits(),
            ],
        )
    # The siblings keep their workers until they finish, so the pool must stay
    # usable and must not deadlock against them.
    result = pool.optimize(config, initial_values, test_functions[0])
    assert result.exit_reason == ExitReason.FINISHED


def test_shared_handler_without_a_pool_aggregates_runs(
    config: Any, test_functions: Any
) -> None:
    history = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[history])
    optimize(config, initial_values, test_functions[0], handlers=[history])
    assert history["results"]


def test_shared_handler_aggregates_across_optimize_many(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any
) -> None:
    single = HistoryHandler()
    optimize(config, initial_values, test_functions[0], handlers=[single])

    shared = HistoryHandler()
    starts = np.tile(initial_values, (3, 1))
    pools(workers=2).optimize_many(
        config,
        starts,
        test_functions[0],
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
    pools: Callable[..., WorkerPool],
    config: Any,
    test_functions: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _mock_scheduler(monkeypatch, _MockedHPCAdapter(tmp_path))
    result = pools(HPCExecutor, workers=2, workdir=tmp_path, template="").evaluate(
        config, initial_values, test_functions[0]
    )
    assert result.results is not None
    assert result.results.target_objective is not None


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_local_jobs_evaluate_through_the_simple_api(
    pools: Callable[..., WorkerPool], config: Any, test_functions: Any, tmp_path: Path
) -> None:
    result = pools(LocalJobExecutor, workers=2, workdir=tmp_path).evaluate(
        config, initial_values, test_functions[0]
    )
    assert result.results is not None
    assert result.results.target_objective is not None


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
    pools: Callable[..., WorkerPool],
) -> None:
    # A minimum of one success is enough to absorb the loss, so without the
    # raise this returns an answer computed from the workers that survived.
    with pytest.raises(ExecutionError, match="could not be run"):
        pools(ProcessExecutor, workers=2).evaluate(
            _DYING_WORKER_CONFIG,
            np.zeros(2),
            _kill_worker_on_one_realization,
        )
