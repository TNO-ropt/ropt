from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.results import FunctionResults
from ropt.workflow import BasicOptimizer
from ropt.workflow.compute_steps import EnsembleEvaluator, EnsembleOptimizer
from ropt.workflow.evaluators import CachedEvaluator, FunctionEvaluator
from ropt.workflow.event_handlers import Observer, Store, Tracker

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult
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
    tracker = Tracker()
    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(tracker)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert tracker["results"] is not None
    assert np.allclose(
        tracker["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
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

    def _function_dict(
        variables: NDArray[np.float64],
        *,
        realization: int,
        test_functions: list[Callable[[NDArray[np.float64], int], float]],
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        return {
            "result": np.fromiter(
                (func(variables, realization) for func in test_functions),
                dtype=np.float64,
            ),
            "foo": "bar",
        }

    evaluator = FunctionEvaluator(
        function=partial(_function_dict, test_functions=test_functions),
        evaluation_info={"foo": object},
    )
    tracker = Tracker()
    step = EnsembleOptimizer(evaluator=evaluator)
    step.add_event_handler(tracker)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert tracker["results"] is not None
    assert np.allclose(
        tracker["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert tracker["results"].evaluations.evaluation_info["foo"] == "bar"


def test_rng(config: dict[str, Any], evaluator: Any) -> None:
    config["variables"]["seed"] = 1
    config2 = deepcopy(config)
    config2["variables"]["seed"] = 2

    tracker = Tracker()
    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(tracker)

    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert tracker["results"] is not None
    variables1 = tracker["results"].evaluations.variables
    tracker["results"] = None
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert tracker["results"] is not None
    variables2 = tracker["results"].evaluations.variables
    tracker["results"] = None
    step.run(variables=initial_values, context=EnOptContext.model_validate(config2))
    assert tracker["results"] is not None
    variables3 = tracker["results"].evaluations.variables

    assert np.all(variables1 == variables2)
    assert not np.all(variables2 == variables3)


def test_set_initial_values(config: dict[str, Any], evaluator: Any) -> None:
    tracker = Tracker()
    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(tracker)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert tracker["results"] is not None
    variables1 = tracker["results"].evaluations.variables
    tracker["variables"] = None
    step.run(
        context=EnOptContext.model_validate(config),
        variables=[0, 0, 0],
    )
    assert tracker["results"] is not None
    variables2 = tracker["results"].evaluations.variables

    assert variables1 is not None
    assert variables2 is not None

    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)

    assert not np.all(variables1 == variables2)


def test_reset_results(config: dict[str, Any], evaluator: Any) -> None:
    tracker = Tracker()
    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(tracker)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    saved_results = deepcopy(tracker["results"])
    tracker["results"] = None
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

    tracker1 = Tracker()
    tracker2 = Tracker(what="last")

    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(tracker1)
    step.add_event_handler(tracker2)
    step.add_event_handler(
        Observer(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    step.run(variables=[0.0, 0.2, 0.1], context=EnOptContext.model_validate(config1))
    assert tracker2["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config2),
        variables=tracker2["results"].evaluations.variables,
    )
    assert tracker2["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config1),
        variables=tracker2["results"].evaluations.variables,
    )
    assert tracker2["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config2),
        variables=tracker2["results"].evaluations.variables,
    )
    assert completed_functions == 14
    assert tracker1["results"] is not None
    assert np.allclose(
        tracker1["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
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

    tracker = Tracker(what="last")
    observer = Observer(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_track_evaluations
    )
    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(tracker)
    step.add_event_handler(observer)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert tracker["results"] is not None
    step.run(
        context=EnOptContext.model_validate(config2),
        variables=tracker["results"].evaluations.variables,
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

    step = EnsembleOptimizer(evaluator=evaluator())
    observer = Observer(
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

    step = EnsembleOptimizer(evaluator=evaluator())
    observer = Observer(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_track_evaluations
    )
    tracker = Tracker(what="last")
    step.add_event_handler(observer)
    step.add_event_handler(tracker)
    for _ in range(2):
        variables = (
            initial_values
            if tracker["results"] is None
            else tracker["results"].evaluations.variables
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

    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(
        Observer(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    tracker = Tracker()
    step.add_event_handler(tracker)
    for _ in range(2):
        variables = (
            initial_values
            if tracker["results"] is None
            else tracker["results"].evaluations.variables
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

    step = EnsembleOptimizer(evaluator=evaluator(new_functions))
    step.add_event_handler(
        Observer(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    tracker = Tracker()
    step.add_event_handler(tracker)
    for _ in range(3):
        variables = (
            initial_values
            if tracker["results"] is None
            else tracker["results"].evaluations.variables
        )
        tracker["results"] = None
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

    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(
        Observer(
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
    tracker = Tracker()
    step = EnsembleEvaluator(evaluator=evaluator())
    step.add_event_handler(tracker)
    step.run(context=EnOptContext.model_validate(config), variables=[0.0, 0.0, 0.1])
    assert tracker["results"].functions is not None
    assert np.allclose(tracker["results"].functions.target_objective, 1.66)

    tracker["results"] = None
    step.run(
        context=EnOptContext.model_validate(config),
        variables=[0, 0, 0],
    )
    assert tracker["results"].functions is not None
    assert np.allclose(tracker["results"].functions.target_objective, 1.75)


def test_evaluator_multi(config: dict[str, Any], evaluator: Any) -> None:
    config["gradient"] = {"evaluation_policy": "speculative"}
    config["optimizer"]["max_functions"] = 4

    store = Store()
    step = EnsembleEvaluator(evaluator=evaluator())
    step.add_event_handler(store)
    step.run(
        context=EnOptContext.model_validate(config),
        variables=np.array([[0, 0, 0.1], [0, 0, 0]]),
    )
    values = [results.functions.target_objective.item() for results in store["results"]]
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

    step = EnsembleOptimizer(evaluator=evaluator())
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
    result_tracker = Tracker()

    def _outer_function(
        variables: NDArray[np.float64],
        perturbation: int,
        **kwargs: Any,  # noqa: ARG001
    ) -> Any:
        new_variables = variables.copy()
        if perturbation < 0:
            new_variables[1] = (
                initial[1]
                if result_tracker["results"] is None
                else result_tracker["results"].evaluations.variables[1]
            )
            tracker = Tracker()
            step = EnsembleOptimizer(evaluator=evaluator())
            step.add_event_handler(tracker)
            step.add_event_handler(result_tracker)
            step.run(
                variables=new_variables,
                context=EnOptContext.model_validate(nested_config),
            )
            return tracker["results"].functions.objectives

        new_variables[1] = result_tracker["results"].evaluations.variables[1]
        return np.fromiter(
            (func(new_variables, 0) for func in test_functions), dtype=np.float64
        )

    outer_evaluator = FunctionEvaluator(function=_outer_function)
    step = EnsembleOptimizer(evaluator=outer_evaluator)
    step.run(variables=initial, context=EnOptContext.model_validate(config))
    assert np.allclose(
        result_tracker["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optimization_abort(config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _observer(_: EnOptEvent) -> None:
        nonlocal last_evaluation

        last_evaluation += 1
        if last_evaluation == 1:
            raise ComputeStepAborted(exit_code=ExitCode.USER_ABORT)

    tracker = Tracker()
    step = EnsembleOptimizer(evaluator=evaluator())
    step.add_event_handler(tracker)
    step.add_event_handler(
        Observer(event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_observer)
    )
    exit_code = step.run(
        variables=initial_values, context=EnOptContext.model_validate(config)
    )
    assert tracker["results"] is not None
    assert exit_code == ExitCode.USER_ABORT
    assert last_evaluation == 1


def _cached_eval(
    obj: CachedEvaluator,
    variables: NDArray[np.float64],
    evaluator_context: EvaluatorContext,
) -> EvaluatorResult:
    results, cached = obj.eval_cached(variables, evaluator_context)
    cached_indices = list(cached.keys())
    info = np.zeros(variables.shape[0], dtype=np.bool_)
    info[cached_indices] = True
    results.evaluation_info = {"cached": info}

    realizations = evaluator_context.realizations.copy()
    realizations[cached_indices] = [item[0] for item in cached.values()]
    names = (
        None
        if evaluator_context.context.names is None
        else evaluator_context.context.names.get("realization")
    )
    if names is not None:
        realizations = np.fromiter((names[idx] for idx in realizations), dtype="U1")
    results.evaluation_info["realizations"] = realizations

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
                    assert np.all(item.evaluations.evaluation_info["cached"])
                else:
                    assert not np.all(item.evaluations.evaluation_info["cached"])
                if names is not None:
                    assert np.all(
                        item.evaluations.evaluation_info["realizations"] == ["a", "b"]
                    )

    config["gradient"] = {"evaluation_policy": "speculative"}
    config["gradient"]["number_of_perturbations"] = "1"
    config["optimizer"]["max_functions"] = 2

    tracker = Tracker(what="last")

    function_evaluator = evaluator((_test_function1, test_functions[1]))
    cached_evaluator = CachedEvaluator(evaluator=function_evaluator, sources={tracker})
    assert isinstance(cached_evaluator, CachedEvaluator)
    monkeypatch.setattr(
        cached_evaluator, "eval", partial(_cached_eval, cached_evaluator)
    )

    step = EnsembleOptimizer(evaluator=cached_evaluator)
    step.add_event_handler(
        Observer(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    step.add_event_handler(tracker)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert completed_test_functions == 8

    completed_test_functions = 0
    step.run(
        context=EnOptContext.model_validate(config),
        variables=tracker["results"].evaluations.variables,
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

    store = Store()
    function_evaluator = evaluator((_test_function1, test_functions[1]))
    cached_evaluator = CachedEvaluator(evaluator=function_evaluator, sources={store})
    step = EnsembleOptimizer(evaluator=cached_evaluator)
    step.add_event_handler(store)

    assert isinstance(cached_evaluator, CachedEvaluator)
    monkeypatch.setattr(
        cached_evaluator, "eval", partial(_cached_eval, cached_evaluator)
    )

    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert completed_test_functions == 4

    completed_test_functions = 0
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    assert completed_test_functions == 2
