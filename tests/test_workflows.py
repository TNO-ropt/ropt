from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.config import EnOptConfig
from ropt.enums import EventType, ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.plugins import PluginManager
from ropt.plugins.evaluator.cached_evaluator import DefaultCachedEvaluator
from ropt.results import FunctionResults
from ropt.workflow import BasicOptimizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult
    from ropt.workflow import Event


initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_run_basic(enopt_config: dict[str, Any], evaluator: Any) -> None:
    plugin_manager = PluginManager()

    tracker = plugin_manager.create_event_handler("tracker")
    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    ).add_event_handler(tracker)
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert tracker["results"] is not None
    assert np.allclose(
        tracker["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_rng(enopt_config: dict[str, Any], evaluator: Any) -> None:
    plugin_manager = PluginManager()
    enopt_config["variables"]["seed"] = 1
    config2 = deepcopy(enopt_config)
    config2["variables"]["seed"] = 2

    tracker = plugin_manager.create_event_handler("tracker")
    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    ).add_event_handler(tracker)

    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert tracker["results"] is not None
    variables1 = tracker["results"].evaluations.variables
    tracker["results"] = None
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert tracker["results"] is not None
    variables2 = tracker["results"].evaluations.variables
    tracker["results"] = None
    step.run(variables=initial_values, config=EnOptConfig.model_validate(config2))
    assert tracker["results"] is not None
    variables3 = tracker["results"].evaluations.variables

    assert np.all(variables1 == variables2)
    assert not np.all(variables2 == variables3)


def test_set_initial_values(enopt_config: dict[str, Any], evaluator: Any) -> None:
    plugin_manager = PluginManager()

    tracker = plugin_manager.create_event_handler("tracker")
    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    ).add_event_handler(tracker)
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert tracker["results"] is not None
    variables1 = tracker["results"].evaluations.variables
    tracker["variables"] = None
    step.run(
        config=EnOptConfig.model_validate(enopt_config),
        transforms=None,
        variables=[0, 0, 0],
    )
    assert tracker["results"] is not None
    variables2 = tracker["results"].evaluations.variables

    assert variables1 is not None
    assert variables2 is not None

    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)

    assert not np.all(variables1 == variables2)


def test_reset_results(enopt_config: dict[str, Any], evaluator: Any) -> None:
    plugin_manager = PluginManager()

    tracker = plugin_manager.create_event_handler("tracker")
    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    ).add_event_handler(tracker)
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    saved_results = deepcopy(tracker["results"])
    tracker["results"] = None
    assert saved_results is not None
    assert np.allclose(saved_results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_two_optimizers_alternating(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    completed_functions = 0

    def _track_evaluations(event: Event) -> None:
        nonlocal completed_functions

        for item in event.data["results"]:
            if isinstance(item, FunctionResults):
                completed_functions += 1

    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    del enopt_config["optimizer"]["tolerance"]

    enopt_config1 = deepcopy(enopt_config)
    enopt_config1["variables"]["mask"] = [True, False, True]
    enopt_config1["optimizer"]["max_functions"] = 4
    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["variables"]["mask"] = [False, True, False]
    enopt_config2["optimizer"]["max_functions"] = 3

    plugin_manager = PluginManager()
    tracker1 = plugin_manager.create_event_handler("tracker")
    tracker2 = plugin_manager.create_event_handler("tracker", what="last")

    step = (
        plugin_manager.create_compute_step(
            "optimizer",
            evaluator=plugin_manager.create_evaluator(
                "function_evaluator", evaluator=evaluator()
            ),
            plugin_manager=plugin_manager,
        )
        .add_event_handler(tracker1)
        .add_event_handler(tracker2)
        .add_event_handler(
            plugin_manager.create_event_handler(
                "observer",
                event_types={EventType.FINISHED_EVALUATION},
                callback=_track_evaluations,
            )
        )
    )
    step.run(
        variables=[0.0, 0.2, 0.1], config=EnOptConfig.model_validate(enopt_config1)
    )
    assert tracker2["results"] is not None
    step.run(
        config=EnOptConfig.model_validate(enopt_config2),
        transforms=None,
        variables=tracker2["results"].evaluations.variables,
    )
    assert tracker2["results"] is not None
    step.run(
        config=EnOptConfig.model_validate(enopt_config1),
        transforms=None,
        variables=tracker2["results"].evaluations.variables,
    )
    assert tracker2["results"] is not None
    step.run(
        config=EnOptConfig.model_validate(enopt_config2),
        transforms=None,
        variables=tracker2["results"].evaluations.variables,
    )
    assert completed_functions == 14
    assert tracker1["results"] is not None
    assert np.allclose(
        tracker1["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optimization_sequential(enopt_config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed

        completed += [
            item for item in event.data["results"] if isinstance(item, FunctionResults)
        ]

    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = 2

    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["optimizer"]["max_functions"] = 3

    plugin_manager = PluginManager()

    tracker = plugin_manager.create_event_handler("tracker", what="last")
    observer = plugin_manager.create_event_handler(
        "observer",
        event_types={EventType.FINISHED_EVALUATION},
        callback=_track_evaluations,
    )
    step = (
        plugin_manager.create_compute_step(
            "optimizer",
            evaluator=plugin_manager.create_evaluator(
                "function_evaluator", evaluator=evaluator()
            ),
            plugin_manager=plugin_manager,
        )
        .add_event_handler(tracker)
        .add_event_handler(observer)
    )
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert tracker["results"] is not None
    step.run(
        config=EnOptConfig.model_validate(enopt_config2),
        transforms=None,
        variables=tracker["results"].evaluations.variables,
    )

    assert not np.allclose(
        completed[1].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert np.all(
        completed[2].evaluations.variables == completed[1].evaluations.variables
    )
    assert np.allclose(completed[-1].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_restart_initial(enopt_config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed

        completed += [
            item for item in event.data["results"] if isinstance(item, FunctionResults)
        ]

    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = 3

    plugin_manager = PluginManager()

    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    )
    observer = plugin_manager.create_event_handler(
        "observer",
        event_types={EventType.FINISHED_EVALUATION},
        callback=_track_evaluations,
    )
    step.add_event_handler(observer)
    for _ in range(2):
        step.run(
            variables=initial_values, config=EnOptConfig.model_validate(enopt_config)
        )

    assert len(completed) == 6
    assert np.all(completed[0].evaluations.variables == initial_values)
    assert np.all(completed[3].evaluations.variables == initial_values)


def test_restart_last(enopt_config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed

        completed += [
            item for item in event.data["results"] if isinstance(item, FunctionResults)
        ]

    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = 3

    plugin_manager = PluginManager()

    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    )
    observer = plugin_manager.create_event_handler(
        "observer",
        event_types={EventType.FINISHED_EVALUATION},
        callback=_track_evaluations,
    )
    tracker = plugin_manager.create_event_handler("tracker", what="last")
    step.add_event_handler(observer).add_event_handler(tracker)
    for _ in range(2):
        variables = (
            initial_values
            if tracker["results"] is None
            else tracker["results"].evaluations.variables
        )
        step.run(
            config=EnOptConfig.model_validate(enopt_config),
            transforms=None,
            variables=variables,
        )

    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )


def test_restart_optimum(enopt_config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed

        completed += [
            item for item in event.data["results"] if isinstance(item, FunctionResults)
        ]

    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = 4

    plugin_manager = PluginManager()

    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    )
    step.add_event_handler(
        plugin_manager.create_event_handler(
            "observer",
            event_types={EventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    tracker = plugin_manager.create_event_handler("tracker")
    step.add_event_handler(tracker)
    for _ in range(2):
        variables = (
            initial_values
            if tracker["results"] is None
            else tracker["results"].evaluations.variables
        )
        step.run(
            config=EnOptConfig.model_validate(enopt_config),
            transforms=None,
            variables=variables,
        )

    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )


def test_restart_optimum_with_reset(
    enopt_config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:
    completed: list[FunctionResults] = []
    max_functions = 5

    def _track_evaluations(event: Event) -> None:
        nonlocal completed

        completed += [
            item for item in event.data["results"] if isinstance(item, FunctionResults)
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

    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = max_functions

    plugin_manager = PluginManager()

    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator(new_functions)
        ),
        plugin_manager=plugin_manager,
    )
    step.add_event_handler(
        plugin_manager.create_event_handler(
            "observer",
            event_types={EventType.FINISHED_EVALUATION},
            callback=_track_evaluations,
        )
    )
    tracker = plugin_manager.create_event_handler("tracker")
    step.add_event_handler(tracker)
    for _ in range(3):
        variables = (
            initial_values
            if tracker["results"] is None
            else tracker["results"].evaluations.variables
        )
        tracker["results"] = None
        step.run(
            config=EnOptConfig.model_validate(enopt_config),
            transforms=None,
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


def test_repeat_metadata(enopt_config: dict[str, Any], evaluator: Any) -> None:
    restarts: list[int] = []

    def _track_results(event: Event) -> None:
        metadata = event.data["results"][0].metadata
        restart = metadata.get("restart", -1)
        assert metadata["foo"] == 1
        assert metadata["bar"] == "string"
        if not restarts or restart != restarts[-1]:
            restarts.append(restart)

    metadata = {
        "foo": 1,
        "bar": "string",
    }

    plugin_manager = PluginManager()

    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    )
    step.add_event_handler(
        plugin_manager.create_event_handler(
            "observer",
            event_types={EventType.FINISHED_EVALUATION},
            callback=_track_results,
        )
    )
    for idx in range(2):
        metadata["restart"] = idx
        step.run(
            config=EnOptConfig.model_validate(enopt_config),
            transforms=None,
            metadata=metadata,
            variables=initial_values,
        )
    assert restarts == [0, 1]


def test_evaluator(enopt_config: dict[str, Any], evaluator: Any) -> None:
    plugin_manager = PluginManager()

    tracker = plugin_manager.create_event_handler("tracker")
    step = plugin_manager.create_compute_step(
        "ensemble_evaluator",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    ).add_event_handler(tracker)
    step.run(config=EnOptConfig.model_validate(enopt_config), variables=[0.0, 0.0, 0.1])
    assert tracker["results"].functions is not None
    assert np.allclose(tracker["results"].functions.weighted_objective, 1.66)

    tracker["results"] = None
    step.run(
        config=EnOptConfig.model_validate(enopt_config),
        transforms=None,
        variables=[0, 0, 0],
    )
    assert tracker["results"].functions is not None
    assert np.allclose(tracker["results"].functions.weighted_objective, 1.75)


def test_evaluator_multi(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = 4

    plugin_manager = PluginManager()

    store = plugin_manager.create_event_handler("store")
    step = plugin_manager.create_compute_step(
        "ensemble_evaluator",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    ).add_event_handler(store)
    step.run(
        config=EnOptConfig.model_validate(enopt_config),
        transforms=None,
        variables=[[0, 0, 0.1], [0, 0, 0]],
    )
    values = [
        results.functions.weighted_objective.item() for results in store["results"]
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
    enopt_config: dict[str, Any],
    evaluator: Any,
    max_criterion: str,
    max_enum: ExitCode,
) -> None:
    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"][max_criterion] = 4

    plugin_manager = PluginManager()

    step = plugin_manager.create_compute_step(
        "optimizer",
        evaluator=plugin_manager.create_evaluator(
            "function_evaluator", evaluator=evaluator()
        ),
        plugin_manager=plugin_manager,
    )
    exit_code = step.run(
        variables=initial_values, config=EnOptConfig.model_validate(enopt_config)
    )
    assert exit_code == max_enum


def test_nested_optimization(enopt_config: dict[str, Any], evaluator: Any) -> None:
    completed_functions = 0

    def _track_evaluations(event: Event) -> None:
        nonlocal completed_functions

        for item in event.data["results"]:
            if isinstance(item, FunctionResults):
                completed_functions += 1

    enopt_config["optimizer"]["tolerance"] = 1e-10
    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = 4
    enopt_config["variables"]["mask"] = [True, False, True]
    nested_config = deepcopy(enopt_config)
    nested_config["variables"]["mask"] = [False, True, False]
    enopt_config["optimizer"]["max_functions"] = 5

    plugin_manager = PluginManager()

    function_evaluator = plugin_manager.create_evaluator(
        "function_evaluator", evaluator=evaluator()
    )
    outer_step = plugin_manager.create_compute_step(
        "optimizer", evaluator=function_evaluator, plugin_manager=plugin_manager
    )
    outer_tracker = plugin_manager.create_event_handler("tracker")
    observer = plugin_manager.create_event_handler(
        "observer",
        event_types={EventType.FINISHED_EVALUATION},
        callback=_track_evaluations,
    )
    outer_step.add_event_handler(observer).add_event_handler(outer_tracker)

    def _inner_optimization(
        variables: NDArray[np.float64],
    ) -> tuple[FunctionResults | None, bool]:
        inner_tracker = plugin_manager.create_event_handler("tracker")
        inner_step = (
            plugin_manager.create_compute_step(
                "optimizer", evaluator=function_evaluator, plugin_manager=plugin_manager
            )
            .add_event_handler(observer)
            .add_event_handler(inner_tracker)
        )
        exit_code = inner_step.run(
            config=EnOptConfig.model_validate(nested_config),
            variables=variables,
        )
        results = inner_tracker["results"]
        assert isinstance(results, FunctionResults | None)
        return results, exit_code == ExitCode.USER_ABORT

    outer_step.run(
        config=EnOptConfig.model_validate(enopt_config),
        transforms=None,
        nested_optimization=_inner_optimization,
        variables=[0.0, 0.2, 0.1],
    )
    assert outer_tracker["results"] is not None
    assert np.allclose(
        outer_tracker["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert completed_functions == 25


def test_nested_optimization_metadata(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    def _track_evaluations(event: Event) -> None:
        for item in event.data["results"]:
            if isinstance(item, FunctionResults):
                assert "inner" in item.metadata or "outer" in item.metadata
                if "outer" in item.metadata:
                    assert item.metadata.get("outer") == 1
                if "inner" in item.metadata:
                    assert item.metadata.get("inner") == "inner_meta_data"

    enopt_config["optimizer"]["tolerance"] = 1e-10
    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["optimizer"]["max_functions"] = 4
    enopt_config["variables"]["mask"] = [True, False, True]
    nested_config = deepcopy(enopt_config)
    nested_config["variables"]["mask"] = [False, True, False]
    enopt_config["optimizer"]["max_functions"] = 5

    plugin_manager = PluginManager()

    function_evaluator = plugin_manager.create_evaluator(
        "function_evaluator", evaluator=evaluator()
    )
    outer_step = plugin_manager.create_compute_step(
        "optimizer", evaluator=function_evaluator, plugin_manager=plugin_manager
    )
    outer_tracker = plugin_manager.create_event_handler("tracker")
    observer = plugin_manager.create_event_handler(
        "observer",
        event_types={EventType.FINISHED_EVALUATION},
        callback=_track_evaluations,
    )
    outer_step.add_event_handler(observer).add_event_handler(outer_tracker)

    def _inner_optimization(
        variables: NDArray[np.float64],
    ) -> tuple[FunctionResults | None, bool]:
        inner_tracker = plugin_manager.create_event_handler("tracker")
        inner_step = (
            plugin_manager.create_compute_step(
                "optimizer", evaluator=function_evaluator, plugin_manager=plugin_manager
            )
            .add_event_handler(observer)
            .add_event_handler(inner_tracker)
        )
        exit_code = inner_step.run(
            config=EnOptConfig.model_validate(nested_config),
            metadata={"inner": "inner_meta_data"},
            variables=variables,
        )
        results = inner_tracker["results"]
        assert isinstance(results, FunctionResults)
        assert results.metadata["inner"] == "inner_meta_data"
        return results, exit_code == ExitCode.USER_ABORT

    outer_step.run(
        config=EnOptConfig.model_validate(enopt_config),
        transforms=None,
        nested_optimization=_inner_optimization,
        metadata={"outer": 1},
        variables=[0.0, 0.2, 0.1],
    )

    assert np.allclose(
        outer_tracker["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert outer_tracker["results"].metadata["outer"] == 1


def test_optimization_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _observer(_: Event) -> None:
        nonlocal last_evaluation

        last_evaluation += 1
        if last_evaluation == 1:
            raise ComputeStepAborted(exit_code=ExitCode.USER_ABORT)

    plugin_manager = PluginManager()

    tracker = plugin_manager.create_event_handler("tracker")
    step = (
        plugin_manager.create_compute_step(
            "optimizer",
            evaluator=plugin_manager.create_evaluator(
                "function_evaluator", evaluator=evaluator()
            ),
            plugin_manager=plugin_manager,
        )
        .add_event_handler(tracker)
        .add_event_handler(
            plugin_manager.create_event_handler(
                "observer",
                event_types={EventType.FINISHED_EVALUATION},
                callback=_observer,
            )
        )
    )
    exit_code = step.run(
        variables=initial_values, config=EnOptConfig.model_validate(enopt_config)
    )
    assert tracker["results"] is not None
    assert exit_code == ExitCode.USER_ABORT
    assert last_evaluation == 1


def _cached_eval(
    obj: DefaultCachedEvaluator,
    variables: NDArray[np.float64],
    context: EvaluatorContext,
) -> EvaluatorResult:
    results, cached = obj.eval_cached(variables, context)
    cached_indices = list(cached.keys())
    info = np.zeros(variables.shape[0], dtype=np.bool_)
    info[cached_indices] = True
    results.evaluation_info = {"cached": info}

    realizations = context.realizations.copy()
    realizations[cached_indices] = [item[0] for item in cached.values()]
    names = (
        None
        if context.config.names is None
        else context.config.names.get("realization")
    )
    if names is not None:
        realizations = np.fromiter((names[idx] for idx in realizations), dtype="U1")
    results.evaluation_info["realizations"] = realizations

    return results


@pytest.mark.parametrize("names", [None, ["a", "b"]])
def test_evaluator_cache(
    enopt_config: dict[str, Any],
    evaluator: Any,
    test_functions: Any,
    monkeypatch: Any,
    names: list[str] | None,
) -> None:
    enopt_config["realizations"] = {"weights": [0.75, 0.25]}
    if names is not None:
        enopt_config["names"] = {"realization": names}

    completed_functions = 0
    completed_test_functions = 0

    def _test_function1(*args: Any, **kwargs: Any) -> float:
        nonlocal completed_test_functions

        completed_test_functions += 1
        return float(test_functions[0](*args, **kwargs))

    def _track_evaluations(event: Event) -> None:
        nonlocal completed_functions

        for item in event.data["results"]:
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

    enopt_config["gradient"] = {"evaluation_policy": "speculative"}
    enopt_config["gradient"]["number_of_perturbations"] = "1"
    enopt_config["optimizer"]["max_functions"] = 2

    plugin_manager = PluginManager()

    tracker = plugin_manager.create_event_handler("tracker", what="last")

    function_evaluator = plugin_manager.create_evaluator(
        "function_evaluator", evaluator=evaluator((_test_function1, test_functions[1]))
    )
    cached_evaluator = plugin_manager.create_evaluator(
        "cached_evaluator", evaluator=function_evaluator, sources={tracker}
    )
    assert isinstance(cached_evaluator, DefaultCachedEvaluator)
    monkeypatch.setattr(
        cached_evaluator, "eval", partial(_cached_eval, cached_evaluator)
    )

    step = (
        plugin_manager.create_compute_step(
            "optimizer", evaluator=cached_evaluator, plugin_manager=plugin_manager
        )
        .add_event_handler(
            plugin_manager.create_event_handler(
                "observer",
                event_types={EventType.FINISHED_EVALUATION},
                callback=_track_evaluations,
            )
        )
        .add_event_handler(tracker)
    )
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert completed_test_functions == 8

    completed_test_functions = 0
    step.run(
        config=EnOptConfig.model_validate(enopt_config),
        variables=tracker["results"].evaluations.variables,
    )
    assert completed_test_functions == 6  # Two evaluations were cached

    assert completed_functions == 4


def test_evaluator_cache_with_store(
    enopt_config: dict[str, Any],
    evaluator: Any,
    test_functions: Any,
    monkeypatch: Any,
) -> None:
    completed_test_functions = 0

    def _test_function1(*args: Any, **kwargs: Any) -> float:
        nonlocal completed_test_functions

        completed_test_functions += 1
        return float(test_functions[0](*args, **kwargs))

    enopt_config["gradient"] = {
        "evaluation_policy": "speculative",
        "number_of_perturbations": "1",
    }
    enopt_config["optimizer"]["max_functions"] = 2

    plugin_manager = PluginManager()

    store = plugin_manager.create_event_handler("store")
    function_evaluator = plugin_manager.create_evaluator(
        "function_evaluator", evaluator=evaluator((_test_function1, test_functions[1]))
    )
    cached_evaluator = plugin_manager.create_evaluator(
        "cached_evaluator", evaluator=function_evaluator, sources={store}
    )
    step = plugin_manager.create_compute_step(
        "optimizer", evaluator=cached_evaluator, plugin_manager=plugin_manager
    ).add_event_handler(store)

    assert isinstance(cached_evaluator, DefaultCachedEvaluator)
    monkeypatch.setattr(
        cached_evaluator, "eval", partial(_cached_eval, cached_evaluator)
    )

    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert completed_test_functions == 4

    completed_test_functions = 0
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    assert completed_test_functions == 2
