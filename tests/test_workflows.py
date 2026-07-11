from __future__ import annotations

import pickle  # noqa: S403
import threading
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, ExitCode
from ropt.evaluation import EvaluationBatchResult
from ropt.exceptions import Abort
from ropt.results import FunctionResults
from ropt.workflow import BasicOptimizer
from ropt.workflow.compute_steps import EvaluationStep, OptimizationStep
from ropt.workflow.evaluators import (
    CachedEvaluator,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    Evaluator,
    FunctionEvaluator,
)
from ropt.workflow.event_handlers import (
    CallbackHandler,
    EventDispatcher,
    EventHandler,
    HistoryHandler,
    ResultsHandler,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt.evaluation import EvaluationBatchContext
    from ropt.events import EnOptEvent


initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 20,
        },
        "backend": {
            "convergence_tolerance": 1e-5,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_run_basic(config: dict[str, Any], evaluator: Any) -> None:
    result_handler = ResultsHandler()
    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(result_handler)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert result_handler["results"] is not None
    assert np.allclose(
        result_handler["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_function_evaluator_with_info(
    config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:

    def _function(
        variables: NDArray[np.float64],
        context: EvaluationFunctionContext,
        *,
        test_functions: list[
            Callable[[NDArray[np.float64], EvaluationFunctionContext], float]
        ],
    ) -> EvaluationFunctionResult:
        return EvaluationFunctionResult(
            objectives=np.fromiter(
                (func(variables, context) for func in test_functions),
                dtype=np.float64,
            ),
            metadata={"foo": "bar"},
        )

    evaluator = FunctionEvaluator(
        function=partial(_function, test_functions=test_functions)
    )
    result_handler = ResultsHandler()
    step = OptimizationStep(evaluator=evaluator)
    step.add_event_handler(result_handler)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert result_handler["results"] is not None
    assert np.allclose(
        result_handler["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert result_handler["results"].evaluations.metadata["foo"] == "bar"


def test_rng(config: dict[str, Any], evaluator: Any) -> None:
    config["variables"]["seed"] = 1
    config2 = deepcopy(config)
    config2["variables"]["seed"] = 2

    result_handler = ResultsHandler()
    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(result_handler)

    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert result_handler["results"] is not None
    variables1 = result_handler["results"].evaluations.variables
    result_handler["results"] = None
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert result_handler["results"] is not None
    variables2 = result_handler["results"].evaluations.variables
    result_handler["results"] = None
    step.run(variables=initial_values, context=EnOptContext.model_validate(config2))
    assert result_handler["results"] is not None
    variables3 = result_handler["results"].evaluations.variables

    assert np.all(variables1 == variables2)
    assert not np.all(variables2 == variables3)


def test_set_initial_values(config: dict[str, Any], evaluator: Any) -> None:
    result_handler = ResultsHandler()
    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(result_handler)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert result_handler["results"] is not None
    variables1 = result_handler["results"].evaluations.variables
    result_handler["variables"] = None
    step.run(
        context=EnOptContext.model_validate(config),
        variables=[0, 0, 0],
    )
    assert result_handler["results"] is not None
    variables2 = result_handler["results"].evaluations.variables

    assert variables1 is not None
    assert variables2 is not None

    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)

    assert not np.all(variables1 == variables2)


def test_reset_results(config: dict[str, Any], evaluator: Any) -> None:
    result_handler = ResultsHandler()
    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(result_handler)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    saved_results = deepcopy(result_handler["results"])
    result_handler["results"] = None
    assert saved_results is not None
    assert np.allclose(saved_results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_two_optimizers_alternating(config: dict[str, Any], evaluator: Any) -> None:
    completed_functions = 0

    def _track_evaluations(event: EnOptEvent) -> None:
        nonlocal completed_functions

        for item in event.results:
            if isinstance(item, FunctionResults):
                completed_functions += 1

    config["gradient"] = {"evaluation_policy": "speculative"}
    del config["backend"]["convergence_tolerance"]

    config1 = deepcopy(config)
    config1["variables"]["mask"] = [True, False, True]
    config1["optimizer"]["max_functions"] = 4
    config2 = deepcopy(config)
    config2["variables"]["mask"] = [False, True, False]
    config2["optimizer"]["max_functions"] = 3

    result_handler1 = ResultsHandler()
    result_handler2 = ResultsHandler(what="last")

    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(result_handler1)
    step.add_event_handler(result_handler2)
    step.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    step.run(variables=[0.0, 0.2, 0.1], context=EnOptContext.model_validate(config1))
    assert result_handler2["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config2),
        variables=result_handler2["results"].evaluations.variables,
    )
    assert result_handler2["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config1),
        variables=result_handler2["results"].evaluations.variables,
    )
    assert result_handler2["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config2),
        variables=result_handler2["results"].evaluations.variables,
    )
    assert completed_functions == 14
    assert result_handler1["results"] is not None
    assert np.allclose(
        result_handler1["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optimization_sequential(config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: EnOptEvent) -> None:
        nonlocal completed

        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    config["gradient"] = {"evaluation_policy": "speculative"}
    config["optimizer"]["max_functions"] = 2

    config2 = deepcopy(config)
    config2["optimizer"]["max_functions"] = 3

    result_handler = ResultsHandler(what="last")
    observer = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_track_evaluations
    )
    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(result_handler)
    step.add_event_handler(observer)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert result_handler["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config2),
        variables=result_handler["results"].evaluations.variables,
    )

    assert not np.allclose(
        completed[1].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert np.all(
        completed[2].evaluations.variables == completed[1].evaluations.variables
    )
    assert np.allclose(completed[-1].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_restart_initial(config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: EnOptEvent) -> None:
        nonlocal completed

        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    config["gradient"] = {"evaluation_policy": "speculative"}
    config["optimizer"]["max_functions"] = 3

    step = OptimizationStep(evaluator=evaluator())
    observer = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_track_evaluations
    )
    step.add_event_handler(observer)
    for _ in range(2):
        step.run(variables=initial_values, context=EnOptContext.model_validate(config))

    assert len(completed) == 6
    assert np.all(completed[0].evaluations.variables == initial_values)
    assert np.all(completed[3].evaluations.variables == initial_values)


def test_restart_last(config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: EnOptEvent) -> None:
        nonlocal completed

        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    config["gradient"] = {"evaluation_policy": "speculative"}
    config["optimizer"]["max_functions"] = 3

    step = OptimizationStep(evaluator=evaluator())
    observer = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_track_evaluations
    )
    result_handler = ResultsHandler(what="last")
    step.add_event_handler(observer)
    step.add_event_handler(result_handler)
    for _ in range(2):
        variables = (
            initial_values
            if result_handler["results"] is None
            else result_handler["results"].evaluations.variables
        )
        step.run(
            context=EnOptContext.model_validate(config),
            variables=variables,
        )

    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )


def test_restart_optimum(config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: EnOptEvent) -> None:
        nonlocal completed

        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    config["gradient"] = {"evaluation_policy": "speculative"}
    config["optimizer"]["max_functions"] = 4

    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    result_handler = ResultsHandler()
    step.add_event_handler(result_handler)
    for _ in range(2):
        variables = (
            initial_values
            if result_handler["results"] is None
            else result_handler["results"].evaluations.variables
        )
        step.run(
            context=EnOptContext.model_validate(config),
            variables=variables,
        )

    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )


def test_restart_optimum_with_reset(
    config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:
    completed: list[FunctionResults] = []
    max_functions = 5

    def _track_evaluations(event: EnOptEvent) -> None:
        nonlocal completed

        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    # Make sure each restart has worse objectives, and that the last evaluation
    # is even worse, so each run has its own optimum that is worse than the
    # global and not at its last evaluation. We should end up with initial
    # values that are not from the global optimum, or from the most recent
    # evaluation:
    new_functions = (
        lambda variables, context: (
            test_functions[0](variables, context)
            + int((len(completed) + 1) / max_functions)
        ),
        lambda variables, context: (
            test_functions[1](variables, context)
            + int((len(completed) + 1) / max_functions)
        ),
    )

    config["gradient"] = {"evaluation_policy": "speculative"}
    config["optimizer"]["max_functions"] = max_functions

    step = OptimizationStep(evaluator=evaluator(new_functions))
    step.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    result_handler = ResultsHandler()
    step.add_event_handler(result_handler)
    for _ in range(3):
        variables = (
            initial_values
            if result_handler["results"] is None
            else result_handler["results"].evaluations.variables
        )
        result_handler["results"] = None
        step.run(
            context=EnOptContext.model_validate(config),
            variables=variables,
        )

    # The third evaluation is the optimum, and used to restart the second run:
    assert np.all(
        completed[max_functions].evaluations.variables
        == completed[2].evaluations.variables
    )
    # The 5th evaluation is the optimum of the second run, and used for the third:
    assert np.all(
        completed[2 * max_functions].evaluations.variables
        == completed[5].evaluations.variables
    )


def test_repeat_metadata(config: dict[str, Any], evaluator: Any) -> None:
    restarts: list[int] = []

    def _track_results(event: EnOptEvent) -> None:
        metadata = event.results[0].metadata
        restart = metadata.get("restart", -1)
        assert metadata["foo"] == 1
        assert metadata["bar"] == "string"
        if not restarts or restart != restarts[-1]:
            restarts.append(restart)

    metadata = {
        "foo": 1,
        "bar": "string",
    }

    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_track_results
        )
    )
    for idx in range(2):
        metadata["restart"] = idx
        step.run(
            context=EnOptContext.model_validate(config),
            metadata=metadata,
            variables=initial_values,
        )
    assert restarts == [0, 1]


def test_evaluator(config: dict[str, Any], evaluator: Any) -> None:
    result_handler = ResultsHandler()
    step = EvaluationStep(evaluator=evaluator())
    step.add_event_handler(result_handler)
    step.run(context=EnOptContext.model_validate(config), variables=[0.0, 0.0, 0.1])
    assert result_handler["results"].functions is not None
    assert np.allclose(result_handler["results"].functions.target_objective, 1.66)

    result_handler["results"] = None
    step.run(
        context=EnOptContext.model_validate(config),
        variables=[0, 0, 0],
    )
    assert result_handler["results"].functions is not None
    assert np.allclose(result_handler["results"].functions.target_objective, 1.75)


def test_evaluator_multi(config: dict[str, Any], evaluator: Any) -> None:
    config["gradient"] = {"evaluation_policy": "speculative"}
    config["optimizer"]["max_functions"] = 4

    history_handler = HistoryHandler()
    step = EvaluationStep(evaluator=evaluator())
    step.add_event_handler(history_handler)
    step.run(
        context=EnOptContext.model_validate(config),
        variables=np.array([[0, 0, 0.1], [0, 0, 0]]),
    )
    values = [
        results.functions.target_objective.item()
        for results in history_handler["results"]
    ]
    assert np.allclose(values, [1.66, 1.75])


@pytest.mark.parametrize(
    ("max_criterion", "max_enum"),
    [
        ("max_functions", ExitCode.MAX_FUNCTIONS_REACHED),
        ("max_batches", ExitCode.MAX_BATCHES_REACHED),
    ],
)
def test_exit_code(
    config: dict[str, Any],
    evaluator: Any,
    max_criterion: str,
    max_enum: ExitCode,
) -> None:
    config["gradient"] = {"evaluation_policy": "speculative"}
    match max_criterion:
        case "max_functions":
            config["optimizer"]["max_functions"] = 4
        case "max_batches":
            config["optimizer"]["max_batches"] = 4

    step = OptimizationStep(evaluator=evaluator())
    exit_code = step.run(
        variables=initial_values, context=EnOptContext.model_validate(config)
    )
    assert exit_code == max_enum


def test_nested_optimization(
    config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:

    config["optimizer"]["max_functions"] = 4
    config["variables"]["mask"] = [True, False, True]
    nested_config = deepcopy(config)
    nested_config["variables"]["mask"] = [False, True, False]

    initial = np.array([0.0, 0.2, 0.1])
    outer_result_handler = ResultsHandler()

    def _optimizer(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> Any:
        new_variables = variables.copy()
        if context.perturbation < 0:
            new_variables[1] = (
                initial[1]
                if outer_result_handler["results"] is None
                else outer_result_handler["results"].evaluations.variables[1]
            )
            result_handler = ResultsHandler()
            step = OptimizationStep(evaluator=evaluator())
            step.add_event_handler(result_handler)
            step.add_event_handler(outer_result_handler)
            step.run(
                variables=new_variables,
                context=EnOptContext.model_validate(nested_config),
            )
            return EvaluationFunctionResult(
                objectives=result_handler["results"].functions.objectives
            )

        new_variables[1] = outer_result_handler["results"].evaluations.variables[1]
        return EvaluationFunctionResult(
            objectives=np.fromiter(
                (func(new_variables, context) for func in test_functions),
                dtype=np.float64,
            )
        )

    outer_evaluator = FunctionEvaluator(function=_optimizer)
    step = OptimizationStep(evaluator=outer_evaluator)
    step.run(variables=initial, context=EnOptContext.model_validate(config))
    assert np.allclose(
        outer_result_handler["results"].evaluations.variables,
        [0.0, 0.0, 0.5],
        atol=0.02,
    )


def test_optimization_abort(config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _observer(_: EnOptEvent) -> None:
        nonlocal last_evaluation

        last_evaluation += 1
        if last_evaluation == 1:
            raise Abort(ExitCode.USER_ABORT)

    result_handler = ResultsHandler()
    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(result_handler)
    step.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_observer
        )
    )
    exit_code = step.run(
        variables=initial_values, context=EnOptContext.model_validate(config)
    )
    assert result_handler["results"] is not None
    assert exit_code == ExitCode.USER_ABORT
    assert last_evaluation == 1


def _cached_eval(
    obj: CachedEvaluator,
    variables: NDArray[np.float64],
    evaluator_context: EvaluationBatchContext,
) -> EvaluationBatchResult:
    results, cached = obj.eval_cached(variables, evaluator_context)
    cached_indices = list(cached.keys())
    realizations = evaluator_context.realizations.copy()
    realizations[cached_indices] = [item[0] for item in cached.values()]
    names = (
        None
        if evaluator_context.context.names is None
        else evaluator_context.context.names.get("realization")
    )
    if names is not None:
        realizations = np.fromiter((names[idx] for idx in realizations), dtype="U1")
    results.metadata["realizations"] = realizations

    return results


@pytest.mark.parametrize("names", [None, ["a", "b"]])
def test_evaluator_cache(
    config: dict[str, Any],
    evaluator: Any,
    test_functions: Any,
    monkeypatch: Any,
    names: list[str] | None,
) -> None:
    config["realizations"] = {"weights": [0.75, 0.25]}
    if names is not None:
        config["names"] = {"realization": names}

    completed_functions = 0
    completed_test_functions = 0

    def _test_function1(*args: Any, **kwargs: Any) -> float:
        nonlocal completed_test_functions

        completed_test_functions += 1
        return float(test_functions[0](*args, **kwargs))

    def _track_evaluations(event: EnOptEvent) -> None:
        nonlocal completed_functions

        for item in event.results:
            if isinstance(item, FunctionResults):
                completed_functions += 1
                if completed_functions == 3:
                    assert np.all(item.evaluations.metadata["cached"])
                else:
                    assert not np.all(item.evaluations.metadata["cached"])
                if names is not None:
                    assert np.all(
                        item.evaluations.metadata["realizations"] == ["a", "b"]
                    )

    config["gradient"] = {"evaluation_policy": "speculative"}
    config["gradient"]["number_of_perturbations"] = "1"
    config["optimizer"]["max_functions"] = 2

    result_handler = ResultsHandler(what="last")

    function_evaluator = evaluator((_test_function1, test_functions[1]))
    cached_evaluator = CachedEvaluator(
        evaluator=function_evaluator, sources={result_handler}, hits_key="cached"
    )
    assert isinstance(cached_evaluator, CachedEvaluator)
    monkeypatch.setattr(
        cached_evaluator, "eval", partial(_cached_eval, cached_evaluator)
    )

    step = OptimizationStep(evaluator=cached_evaluator)
    step.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    step.add_event_handler(result_handler)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert completed_test_functions == 8

    completed_test_functions = 0
    step.run(
        context=EnOptContext.model_validate(config),
        variables=result_handler["results"].evaluations.variables,
    )
    assert completed_test_functions == 6  # Two evaluations were cached

    assert completed_functions == 4


def test_evaluator_cache_with_store(
    config: dict[str, Any],
    evaluator: Any,
    test_functions: Any,
    monkeypatch: Any,
) -> None:
    completed_test_functions = 0

    def _test_function1(*args: Any, **kwargs: Any) -> float:
        nonlocal completed_test_functions

        completed_test_functions += 1
        return float(test_functions[0](*args, **kwargs))

    config["gradient"] = {
        "evaluation_policy": "speculative",
        "number_of_perturbations": "1",
    }
    config["optimizer"]["max_functions"] = 2

    history_handler = HistoryHandler()
    function_evaluator = evaluator((_test_function1, test_functions[1]))
    cached_evaluator = CachedEvaluator(
        evaluator=function_evaluator, sources={history_handler}
    )
    step = OptimizationStep(evaluator=cached_evaluator)
    step.add_event_handler(history_handler)

    assert isinstance(cached_evaluator, CachedEvaluator)
    monkeypatch.setattr(
        cached_evaluator, "eval", partial(_cached_eval, cached_evaluator)
    )

    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert completed_test_functions == 4

    completed_test_functions = 0
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert completed_test_functions == 2


def test_evaluator_raises_on_concurrent_use() -> None:
    in_eval = threading.Event()
    can_finish = threading.Event()
    thread2_error: list[BaseException | None] = [None]

    class _BlockingEvaluator(Evaluator):
        def eval(self, _variables: Any, _context: Any) -> EvaluationBatchResult:  # noqa: PLR6301
            in_eval.set()
            can_finish.wait()
            return EvaluationBatchResult(objectives=np.zeros((1, 1), dtype=np.float64))

    ev = _BlockingEvaluator()

    def _thread1() -> None:
        ev.eval(None, None)

    def _thread2() -> None:
        in_eval.wait()
        try:
            ev.eval(None, None)
        except RuntimeError as exc:
            thread2_error[0] = exc

    t1 = threading.Thread(target=_thread1)
    t2 = threading.Thread(target=_thread2)
    t1.start()
    t2.start()
    t2.join(timeout=5.0)
    can_finish.set()
    t1.join(timeout=5.0)

    assert isinstance(thread2_error[0], RuntimeError)
    assert "thread" in str(thread2_error[0])


class _RecordingEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
        self.threads: list[int] = []

    def eval(self, _variables: Any, _context: Any) -> EvaluationBatchResult:
        self.threads.append(threading.get_ident())
        return EvaluationBatchResult(objectives=np.zeros((1, 1), dtype=np.float64))


def test_evaluator_pins_to_first_thread_when_staggered() -> None:
    evaluator = _RecordingEvaluator()

    # First use finishes on a worker thread; a later call from a different
    # thread is rejected even though the two calls do not overlap.
    _run_in_thread(lambda: evaluator.eval(None, None))
    with pytest.raises(RuntimeError, match="thread"):
        evaluator.eval(None, None)


def test_evaluator_allows_repeated_use_on_same_thread() -> None:
    evaluator = _RecordingEvaluator()

    evaluator.eval(None, None)
    evaluator.eval(None, None)

    assert len(evaluator.threads) == 2


def test_evaluator_pickle_before_use_succeeds_and_resets_owner() -> None:
    evaluator = _RecordingEvaluator()

    restored = pickle.loads(pickle.dumps(evaluator))  # noqa: S301

    # Usable after unpickling (its lock was recreated) and pins to the first
    # thread that uses it -- here the main thread, which stays alive.
    restored.eval(None, None)
    assert len(restored.threads) == 1

    errors: list[BaseException] = []

    def _use() -> None:
        try:
            restored.eval(None, None)
        except RuntimeError as exc:
            errors.append(exc)

    _run_in_thread(_use)

    assert len(errors) == 1
    assert "thread" in str(errors[0])


def test_evaluator_pickle_after_use_raises() -> None:
    evaluator = _RecordingEvaluator()
    evaluator.eval(None, None)
    with pytest.raises(RuntimeError, match="after it has been used"):
        pickle.dumps(evaluator)


def test_event_handler_raises_on_concurrent_use() -> None:
    in_handle = threading.Event()
    can_finish = threading.Event()
    thread2_error: list[BaseException | None] = [None]

    class _BlockingHandler(EventHandler):
        @property
        def event_types(self) -> set[EnOptEventType]:
            return {EnOptEventType.FINISHED_EVALUATION}

        def handle_event(self, _event: EnOptEvent) -> None:  # noqa: PLR6301
            in_handle.set()
            can_finish.wait()

    handler = _BlockingHandler()
    mock_event = object()

    def _thread1() -> None:
        handler.handle_event(mock_event)  # type: ignore[arg-type]

    def _thread2() -> None:
        in_handle.wait()
        try:
            handler.handle_event(mock_event)  # type: ignore[arg-type]
        except RuntimeError as exc:
            thread2_error[0] = exc

    t1 = threading.Thread(target=_thread1)
    t2 = threading.Thread(target=_thread2)
    t1.start()
    t2.start()
    t2.join(timeout=5.0)
    can_finish.set()
    t1.join(timeout=5.0)

    assert isinstance(thread2_error[0], RuntimeError)
    assert "thread" in str(thread2_error[0])


class _RecordingHandler(EventHandler):
    def __init__(self) -> None:
        super().__init__()
        self.threads: list[int] = []

    @property
    def event_types(self) -> set[EnOptEventType]:
        return {EnOptEventType.FINISHED_EVALUATION}

    def handle_event(self, _event: EnOptEvent) -> None:
        self.threads.append(threading.get_ident())


def _run_in_thread(target: Callable[[], object]) -> None:
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=5.0)


def test_event_handler_pins_to_first_thread_when_staggered() -> None:
    handler = _RecordingHandler()
    event = object()

    # First use happens on a worker thread and finishes completely.
    _run_in_thread(lambda: handler.handle_event(event))  # type: ignore[arg-type]

    # A later, non-overlapping call from a different thread is still rejected.
    with pytest.raises(RuntimeError, match="thread"):
        handler.handle_event(event)  # type: ignore[arg-type]


def test_event_handler_allows_repeated_use_on_same_thread() -> None:
    handler = _RecordingHandler()
    event = object()

    handler.handle_event(event)  # type: ignore[arg-type]
    handler.handle_event(event)  # type: ignore[arg-type]

    assert len(handler.threads) == 2


def test_dispatcher_handler_is_not_thread_pinned() -> None:
    handler = _RecordingHandler()
    handler.register_dispatcher()
    event = object()
    errors: list[BaseException] = []

    def _use() -> None:
        try:
            handler.handle_event(event)  # type: ignore[arg-type]
        except RuntimeError as exc:  # pragma: no cover - should not happen
            errors.append(exc)

    _run_in_thread(_use)
    _run_in_thread(_use)

    assert errors == []
    assert len(handler.threads) == 2


def test_register_dispatcher_twice_raises() -> None:
    handler = _RecordingHandler()
    handler.register_dispatcher()
    with pytest.raises(RuntimeError, match="already registered with a dispatcher"):
        handler.register_dispatcher()


def test_register_dispatcher_via_add_event_handler_twice_raises() -> None:
    handler = _RecordingHandler()
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(handler)
    with pytest.raises(RuntimeError, match="already registered with a dispatcher"):
        dispatcher.add_event_handler(handler)


def test_dispatcher_then_compute_step_raises() -> None:
    handler = _RecordingHandler()
    handler.register_dispatcher()
    with pytest.raises(RuntimeError, match="registered with a dispatcher"):
        handler.register_compute_step()


def test_compute_step_then_dispatcher_raises() -> None:
    handler = _RecordingHandler()
    handler.register_compute_step()
    with pytest.raises(RuntimeError, match="compute step"):
        handler.register_dispatcher()


def test_handler_may_be_attached_to_multiple_compute_steps() -> None:
    handler = _RecordingHandler()
    handler.register_compute_step()
    handler.register_compute_step()  # allowed, no error


def test_pickle_before_use_succeeds_and_resets_owner() -> None:
    handler = _RecordingHandler()

    restored = pickle.loads(pickle.dumps(handler))  # noqa: S301

    # Usable after unpickling (its lock was recreated) and pins to the first
    # thread that uses it -- here the main thread, which stays alive.
    restored.handle_event(object())
    assert len(restored.threads) == 1

    # A different, concurrently-live thread is rejected.
    errors: list[BaseException] = []

    def _use() -> None:
        try:
            restored.handle_event(object())
        except RuntimeError as exc:
            errors.append(exc)

    _run_in_thread(_use)

    assert len(errors) == 1
    assert "thread" in str(errors[0])


def test_pickle_after_use_raises() -> None:
    handler = _RecordingHandler()
    handler.handle_event(object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="after it has been used"):
        pickle.dumps(handler)
