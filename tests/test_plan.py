from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted, PlanAborted
from ropt.plan import (
    BasicOptimizer,
    Event,
    OptimizerContext,
    Plan,
)
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ruff: noqa: SLF001


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_run_basic(enopt_config: dict[str, Any], evaluator: Any) -> None:
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="opt")
    step = plan.add_step("optimizer", tags="opt")
    plan.run_step(step, config=enopt_config)
    assert tracker["results"] is not None
    assert np.allclose(tracker["variables"], [0.0, 0.0, 0.5], atol=0.02)

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_plan_rng(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["gradient"]["seed"] = 1
    config2 = deepcopy(enopt_config)
    config2["gradient"]["seed"] = 2

    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    tracker1 = plan.add_handler("tracker", tags="optimal1")
    tracker2 = plan.add_handler("tracker", tags="optimal2")
    tracker3 = plan.add_handler("tracker", tags="optimal3")
    step1 = plan.add_step("optimizer", tags="optimal1")
    step2 = plan.add_step("optimizer", tags="optimal2")
    step3 = plan.add_step("optimizer", tags="optimal3")
    plan.run_step(step1, config=enopt_config)
    plan.run_step(step2, config=enopt_config)
    plan.run_step(step3, config=config2)

    assert tracker1["results"] is not None
    assert tracker2["results"] is not None
    assert tracker3["results"] is not None

    assert np.all(tracker1["variables"] == tracker2["variables"])
    assert not np.all(tracker1["variables"] == tracker3["variables"])


def test_set_initial_values(enopt_config: dict[str, Any], evaluator: Any) -> None:
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="optimal")
    step = plan.add_step("optimizer", tags="optimal")
    plan.run_step(step, config=enopt_config)
    variables1 = tracker["variables"]
    tracker["variables"] = None
    plan.run_step(step, config=enopt_config, variables=[0, 0, 0])
    variables2 = tracker["variables"]

    assert variables1 is not None
    assert variables2 is not None

    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)

    assert not np.all(variables1 == variables2)


def test_reset_results(enopt_config: dict[str, Any], evaluator: Any) -> None:
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="opt")
    step = plan.add_step("optimizer", tags="opt")
    plan.run_step(step, config=enopt_config)
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

    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    opt_config1 = {
        "speculative": True,
        "max_functions": 4,
    }
    opt_config2 = {
        "speculative": True,
        "max_functions": 3,
    }

    enopt_config1 = deepcopy(enopt_config)
    enopt_config1["variables"]["mask"] = [True, False, True]
    enopt_config1["optimizer"] = opt_config1
    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["variables"]["mask"] = [False, True, False]
    enopt_config2["optimizer"] = opt_config2

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    plan = Plan(context)
    tracker1 = plan.add_handler("tracker", tags="opt")
    tracker2 = plan.add_handler("tracker", tags="opt", what="last")
    step1 = plan.add_step("optimizer", tags="opt")
    step2 = plan.add_step("optimizer", tags="opt")
    plan.run_step(step1, config=enopt_config1)
    plan.run_step(step2, config=enopt_config2, variables=tracker2["variables"])
    plan.run_step(step1, config=enopt_config1, variables=tracker2["variables"])
    plan.run_step(step2, config=enopt_config2, variables=tracker2["variables"])
    assert completed_functions == 14
    assert tracker1["results"] is not None
    assert np.allclose(tracker1["variables"], [0.0, 0.0, 0.5], atol=0.02)


def test_optimization_sequential(enopt_config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed

        completed += [
            item for item in event.data["results"] if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 2

    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["optimizer"]["max_functions"] = 3

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="last", what="last")
    step1 = plan.add_step("optimizer", tags="last")
    step2 = plan.add_step("optimizer", tags="last")
    plan.run_step(step1, config=enopt_config)
    plan.run_step(step2, config=enopt_config2, variables=tracker["variables"])

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

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    plan = Plan(context)
    step = plan.add_step("optimizer")
    for _ in range(2):
        plan.run_step(step, config=enopt_config)

    assert len(completed) == 6
    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)


def test_restart_last(enopt_config: dict[str, Any], evaluator: Any) -> None:
    completed: list[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed

        completed += [
            item for item in event.data["results"] if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="last", what="last")
    step = plan.add_step("optimizer", tags="last")
    for _ in range(2):
        plan.run_step(step, config=enopt_config, variables=tracker["variables"])

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

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="optimum")
    step = plan.add_step("optimizer", tags="optimum")
    for _ in range(2):
        plan.run_step(step, config=enopt_config, variables=tracker["variables"])

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

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = max_functions

    context = OptimizerContext(evaluator=evaluator(new_functions)).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="opt")
    step = plan.add_step("optimizer", tags="opt")
    for _ in range(3):
        initial = tracker["variables"]
        tracker["results"] = None
        plan.run_step(step, config=enopt_config, variables=initial)

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

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_results
    )
    plan = Plan(context)
    step = plan.add_step("optimizer", tags="opt")
    for idx in range(2):
        metadata["restart"] = idx
        plan.run_step(step, config=enopt_config, metadata=metadata)
    assert restarts == [0, 1]


def test_evaluator_step(enopt_config: dict[str, Any], evaluator: Any) -> None:
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="eval")
    step = plan.add_step("evaluator", tags="eval")
    plan.run_step(step, config=enopt_config)
    assert tracker["results"].functions is not None
    assert np.allclose(tracker["results"].functions.weighted_objective, 1.66)

    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="eval")
    step = plan.add_step("evaluator", tags="eval")
    plan.run_step(step, config=enopt_config, variables=[0, 0, 0])
    assert tracker["results"].functions is not None
    assert np.allclose(tracker["results"].functions.weighted_objective, 1.75)


def test_evaluator_step_multi(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="eval", what="all")
    step = plan.add_step("evaluator", tags="eval")
    plan.run_step(step, config=enopt_config, variables=[[0, 0, 0.1], [0, 0, 0]])
    values = [
        results.functions.weighted_objective.item() for results in tracker["results"]
    ]
    assert np.allclose(values, [1.66, 1.75])


def test_exit_code(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(context)
    step = plan.add_step("optimizer")
    exit_code = plan.run_step(step, config=enopt_config)
    assert exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED


def test_nested_plan(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    completed_functions = 0

    def _track_evaluations(event: Event) -> None:
        nonlocal completed_functions

        for item in event.data["results"]:
            if isinstance(item, FunctionResults):
                completed_functions += 1

    enopt_config["optimizer"]["tolerance"] = 1e-10
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4
    enopt_config["variables"]["mask"] = [True, False, True]
    nested_config = deepcopy(enopt_config)
    nested_config["variables"]["mask"] = [False, True, False]
    enopt_config["optimizer"]["max_functions"] = 5

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )

    inner_plan = Plan(context)
    inner_tracker = inner_plan.add_handler("tracker", tags="inner")
    inner_step = inner_plan.add_step("optimizer", tags="inner")

    def _inner_optimization(
        plan: Plan, variables: NDArray[np.float64]
    ) -> FunctionResults | None:
        plan.run_step(inner_step, config=nested_config, variables=variables)
        results = inner_tracker["results"]
        assert isinstance(results, FunctionResults | None)
        return results

    inner_plan.add_function(_inner_optimization)

    outer_plan = Plan(context)
    outer_tracker = outer_plan.add_handler("tracker", tags="outer")
    outer_step = outer_plan.add_step("optimizer", tags="outer")
    outer_plan.run_step(outer_step, config=enopt_config, nested_optimization=inner_plan)
    assert outer_tracker["results"] is not None
    assert np.allclose(outer_tracker["variables"], [0.0, 0.0, 0.5], atol=0.02)
    assert completed_functions == 25


def test_nested_plan_metadata(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    def _track_evaluations(event: Event) -> None:
        for item in event.data["results"]:
            if isinstance(item, FunctionResults):
                if "outer" in event.tags:
                    assert item.metadata.get("outer") == 1
                if "inner" in event.tags:
                    assert item.metadata.get("inner") == "inner_meta_data"

    enopt_config["optimizer"]["tolerance"] = 1e-10
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4
    enopt_config["variables"]["mask"] = [True, False, True]
    nested_config = deepcopy(enopt_config)
    nested_config["variables"]["mask"] = [False, True, False]
    enopt_config["optimizer"]["max_functions"] = 5

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )

    inner_plan = Plan(context)
    inner_tracker = inner_plan.add_handler("tracker", tags="inner")
    inner_step = inner_plan.add_step("optimizer", tags="inner")

    def _inner_optimization(
        plan: Plan, variables: NDArray[np.float64]
    ) -> FunctionResults | None:
        plan.run_step(
            inner_step,
            config=nested_config,
            metadata={"inner": "inner_meta_data"},
            variables=variables,
        )
        results = inner_tracker["results"]
        assert isinstance(results, FunctionResults)
        return results

    inner_plan.add_function(_inner_optimization)

    outer_plan = Plan(context)
    outer_tracker = outer_plan.add_handler("tracker", tags="outer")
    outer_step = outer_plan.add_step("optimizer", tags="outer")
    outer_plan.run_step(
        outer_step,
        config=enopt_config,
        nested_optimization=inner_plan,
        metadata={"outer": 1},
    )

    assert inner_tracker["results"] is not None
    assert np.allclose(outer_tracker["variables"], [0.0, 0.0, 0.5], atol=0.02)
    assert outer_tracker["results"].metadata["outer"] == 1
    assert inner_tracker["results"].metadata["inner"] == "inner_meta_data"


def test_plan_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _observer(_: Event) -> None:
        nonlocal last_evaluation

        last_evaluation += 1
        if last_evaluation == 1:
            raise OptimizationAborted(exit_code=OptimizerExitCode.USER_ABORT)

    context = OptimizerContext(evaluator=evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _observer
    )
    plan = Plan(context)
    tracker = plan.add_handler("tracker", tags="opt")
    step = plan.add_step("optimizer", tags="opt")
    plan.run_step(step, config=enopt_config)
    assert tracker["results"] is not None
    assert last_evaluation == 1
    assert plan.aborted

    step = plan.add_step("optimizer", tags="opt")
    with pytest.raises(PlanAborted, match="Plan was aborted by the previous step."):
        plan.run_step(step, config=enopt_config)
