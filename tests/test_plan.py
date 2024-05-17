from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from ropt.enums import EventType
from ropt.optimization import EnsembleOptimizer, Plan, PlanContext
from ropt.plugins import PluginManager
from ropt.plugins.optimization_steps.base import (
    OptimizationSteps,
    OptimizationStepsPlugin,
)
from ropt.plugins.optimization_steps.evaluator import DefaultEvaluatorStep
from ropt.report import ResultsDataFrame
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.events import OptimizationEvent


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_restart_step(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4
    optimizer = EnsembleOptimizer(evaluator())
    results1 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results1 is not None
    assert np.allclose(results1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    optimizer = EnsembleOptimizer(evaluator())
    results2 = optimizer.start_optimization(
        plan=[
            {"tracker": {"id": "optimum", "source": "opt"}},
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"restart": {"max_restarts": 0}},
        ],
    )
    assert results2 is not None
    assert np.all(results1.evaluations.variables == results2.evaluations.variables)


def test_restart_initial(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1}},
        ],
    )

    assert len(completed) == 6

    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)


def test_restart_not_at_the_end(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1}},
            {"optimizer": {}},
        ],
    )

    assert len(completed) == 9

    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)
    assert np.all(completed[6].evaluations.variables == initial)


def test_restart_label(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"label": "restart_here"},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1, "label": "restart_here"}},
        ],
    )

    assert len(completed) == 9


def test_restart_last(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True

    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"update_config": {"initial_variables": "last"}},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"source": "opt", "type": "last_result", "id": "last"}},
            {"restart": {"max_restarts": 1}},
        ],
    )

    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )


def test_restart_optimal(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"tracker": {"id": "optimum", "source": "opt"}},
            {"config": enopt_config},
            {"update_config": {"initial_variables": "optimum"}},
            {"optimizer": {"id": "opt"}},
            {"restart": {"max_restarts": 1}},
        ],
    )

    # The third evaluation is the optimum, and used to restart:
    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )


def test_restart_optimal_step(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    completed: List[FunctionResults] = []
    max_functions = 5

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
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

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = max_functions

    optimizer = EnsembleOptimizer(evaluator(new_functions))
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"tracker": {"id": "optimum", "source": "opt"}},
            {"update_config": {"initial_variables": "optimum"}},
            {"optimizer": {"id": "opt"}},
            {"reset_tracker": "optimum"},
            {"restart": {"max_restarts": 2}},
        ],
    )

    # The third evaluation is the optimum, and used to restart the second run:
    assert np.all(
        completed[max_functions].evaluations.variables
        == completed[2].evaluations.variables
    )
    # The 8th evaluation is the optimum of the second run, and used for the third:
    assert np.all(
        completed[2 * max_functions].evaluations.variables
        == completed[8].evaluations.variables
    )


def test_restart_metadata(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(
        {"result_id", "evaluations.variables", "metadata.restart"}
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        lambda event: reporter.add_results(event.config, event.results),  # type: ignore
    )
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {}},
            {"restart": {"max_restarts": 1}},
        ]
    )

    assert reporter.frame["metadata.restart"].to_list() == 3 * [0] + 3 * [1]


class ModifyConfigStep:
    def __init__(
        self,
        _0: Dict[str, Any],
        _1: PlanContext,
        plan: Plan,
        weights: NDArray[np.float64],
    ) -> None:
        self._plan = plan
        self._weights = weights

    def run(self, _: Optional[NDArray[np.float64]]) -> bool:
        self._plan.update_enopt_config(
            {"objective_functions": {"weights": self._weights}}
        )
        return False


class ModifyConfig(OptimizationSteps):
    def __init__(self, context: PlanContext, plan: Plan) -> None:
        self._context = context
        self._plan = plan

    def get_step(self, config: Dict[str, Any]) -> Any:
        weights = config["modify"]["weights"]
        return ModifyConfigStep(config, self._context, self._plan, weights)


class ModifyConfigPlugin(OptimizationStepsPlugin):
    def create(self, context: PlanContext, plan: Plan) -> ModifyConfig:
        return ModifyConfig(context, plan)

    def is_supported(self, method: str) -> bool:
        return method in {"modify"}


def test_modify_enopt_in_plan(enopt_config: Any, evaluator: Any) -> None:
    weights = enopt_config["objective_functions"]["weights"]
    enopt_config["objective_functions"]["weights"] = [1, 1]
    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert not np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    plugin_manager = PluginManager()
    plugin_manager.add_plugins(
        "optimization_step",
        {"modify_plugin": ModifyConfigPlugin()},
    )

    optimizer = EnsembleOptimizer(evaluator(), plugin_manager=plugin_manager)
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"modify": {"weights": weights}},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


class EvaluatorWithProcessStep(DefaultEvaluatorStep):
    def __init__(
        self,
        config: Dict[str, Any],
        context: PlanContext,
        plan: Plan,
    ) -> None:
        super().__init__(config, context, plan)
        self._completed = config["completed"]

    def process(self, results: FunctionResults) -> None:
        self._completed.append(results)


class EvaluatorWithProcess((OptimizationSteps)):
    def __init__(self, context: PlanContext, plan: Plan) -> None:
        self._context = context
        self._plan = plan

    def get_step(self, config: Dict[str, Any]) -> Any:
        return EvaluatorWithProcessStep(
            config["evaluator_with_process"], self._context, self._plan
        )


class EvaluatorWithProcessPlugin(OptimizationStepsPlugin):
    def create(self, context: PlanContext, plan: Plan) -> EvaluatorWithProcess:
        return EvaluatorWithProcess(context, plan)

    def is_supported(self, _: str) -> bool:
        return True


@pytest.mark.parametrize("method", ["evaluator", "evaluator_with_process"])
@pytest.mark.parametrize("init_from", [None, "last", "optimum"])
def test_evaluator_step(
    enopt_config: Any, evaluator: Any, init_from: str, method: str
) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    plugin_manager = PluginManager()
    if method == "evaluator_with_process":
        plugin_manager.add_plugins(
            "optimization_step",
            {"evaluator_plugin": EvaluatorWithProcessPlugin()},
        )

    optimizer = EnsembleOptimizer(evaluator(), plugin_manager=plugin_manager)
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    if method == "evaluator":
        optimizer.add_observer(EventType.FINISHED_EVALUATOR_STEP, _track_evaluations)

    plan = [
        {"config": enopt_config},
        {"optimizer": {"id": "opt"}},
    ]
    if init_from is None:
        plan.append({method: {"completed": completed}})
    elif init_from == "last":
        plan.extend(
            [
                {"tracker": {"id": "last", "source": "opt", "type": "last_result"}},
                {"update_config": {"initial_variables": "last"}},
                {method: {"completed": completed}},
            ]
        )
    else:
        plan.extend(
            [
                {"update_config": {"initial_variables": "optimum"}},
                {method: {"completed": completed}},
            ]
        )
    plan.append({"tracker": {"id": "optimum", "source": "opt"}})
    results = optimizer.start_optimization(plan=plan)

    from_map = {None: 0, "last": 3, "optimum": 2}

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert results.result_id == 2 * from_map["optimum"]
    assert np.all(
        completed[from_map[init_from]].evaluations.variables
        == completed[-1].evaluations.variables
    )


def test_restart_last_optimum_from_evaluation(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["optimizer"]["speculative"] = True

    completed: List[Tuple[EventType, float]] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        assert event.results is not None
        nonlocal completed
        completed += [
            (event.event_type, float(item.functions.weighted_objective))
            for item in event.results
            if isinstance(item, FunctionResults) and item.functions is not None
        ]

    enopt_config["optimizer"]["max_functions"] = 3

    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    optimizer.add_observer(EventType.FINISHED_EVALUATOR_STEP, _track_evaluations)
    optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"tracker": {"id": "eval_tracker", "source": "eval"}},
            {"tracker": {"id": "opt_tracker", "source": "opt"}},
            {"update_config": {"initial_variables": "eval_tracker"}},
            {"optimizer": {"id": "opt"}},
            {"update_config": {"initial_variables": "opt_tracker"}},
            {"evaluator": {"id": "eval"}},
            {"reset_tracker": "opt_tracker"},
            {"restart": {"max_restarts": 3}},
        ]
    )

    min_opt = min(
        item[1] for item in completed if item[0] == EventType.FINISHED_EVALUATION
    )
    min_eval = min(
        item[1] for item in completed if (item[0] == EventType.FINISHED_EVALUATOR_STEP)
    )
    assert optimizer.results["eval_tracker"] is not None
    assert optimizer.results["eval_tracker"].functions is not None
    assert optimizer.results["eval_tracker"].functions.weighted_objective == min_eval
    assert optimizer.results["opt_tracker"] is not None
    assert optimizer.results["opt_tracker"].functions is not None
    assert optimizer.results["opt_tracker"].functions.weighted_objective == min_eval

    assert min_opt == min_eval
