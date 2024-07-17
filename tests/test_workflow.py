from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.config.workflow import WorkflowConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.events import OptimizationEvent
from ropt.exceptions import WorkflowError
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizationWorkflow, OptimizerContext, Workflow

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig

# ruff: noqa: SLF001


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


def test_run_basic(enopt_config: Any, evaluator: Any) -> None:
    variables = BasicOptimizationWorkflow(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_invalid_context_ids() -> None:
    workflow_config: Dict[str, Any] = {
        "context": [
            {
                "id": "1optimal",
                "init": "tracker",
            },
        ],
        "steps": [],
    }
    with pytest.raises(ValidationError, match=".*Invalid ID: 1optimal.*"):
        WorkflowConfig.model_validate(workflow_config)


def test_duplicate_context_ids() -> None:
    workflow_config: Dict[str, Any] = {
        "context": [
            {
                "id": "optimal",
                "init": "tracker",
            },
            {
                "id": "optimal",
                "init": "tracker",
            },
        ],
        "steps": [],
    }
    with pytest.raises(
        ValidationError, match=".*Duplicate Context ID\\(s\\): optimal.*"
    ):
        WorkflowConfig.model_validate(workflow_config)


def test_parse_value(enopt_config: Any, evaluator: Any) -> None:
    workflow_config: Dict[str, Any] = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "results",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["results"],
                },
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    assert workflow.parse_value("${{ 1 }}") == 1
    assert workflow.parse_value("${{ -1 }}") == -1
    assert not workflow.parse_value("${{ not 1 }}")
    assert not workflow.parse_value("${{ True and False }}")
    assert workflow.parse_value("${{ True or False }}")
    assert workflow.parse_value("${{ 1 + 1 }}") == 2
    assert workflow.parse_value("${{ 2**3 }}") == 8
    assert workflow.parse_value("${{ 3 % 2 }}") == 1
    assert workflow.parse_value("${{ 3 // 2 }}") == 1
    assert workflow.parse_value("${{ 2.5 + (2 + 3) / 2 }}") == 5
    assert workflow.parse_value("${{ 1 < 2 }}")
    assert workflow.parse_value("${{ 1 < 2 < 3 }}")
    assert not workflow.parse_value("${{ 1 < 2 > 3 }}")

    assert workflow.parse_value("$results") is None
    assert workflow.parse_value("${{ [1, 2] }}") == [1, 2]
    assert workflow.parse_value("${{ [$results, 2] }}") == [None, 2]

    assert workflow.parse_value("a ${{ 1 }} b") == "a 1 b"
    assert workflow.parse_value("a ${{ 1 + 1 }} b") == "a 2 b"
    assert workflow.parse_value("a ${{ 1 + 1 }} b $results") == "a 2 b None"
    assert workflow.parse_value("a ${{ 1 + 1 }} b $$results") == "a 2 b $results"

    with pytest.raises(
        WorkflowError,
        match=re.escape("Syntax error in workflow expression: 1 + 1 ${{ x"),
    ):
        workflow.parse_value("a $results ${{ 1 + 1 ${{ x }} }} b")

    with pytest.raises(
        WorkflowError, match=re.escape("Syntax error in workflow expression: 1 + * 1")
    ):
        workflow.parse_value("${{ 1 + * 1 }}")

    with pytest.raises(
        WorkflowError,
        match=re.escape("Unknown workflow variable: `y`"),
    ):
        workflow.parse_value("${{ $y + 1 }}")

    workflow.run()

    assert isinstance(workflow.parse_value("$results"), Results)


def test_setvar(evaluator: Any) -> None:
    workflow_config: Dict[str, Any] = {
        "steps": [
            {
                "run": "setvar",
                "with": {
                    "var": "x",
                    "value": "${{ 1 }}",
                },
            },
            {
                "run": "setvar",
                "with": {
                    "var": "y",
                    "value": 1,
                },
            },
            {
                "run": "setvar",
                "with": {
                    "expr": "z = $y + 1",
                },
            },
            {
                "run": "setvar",
                "with": "u = 1",
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()
    assert workflow["x"] == 1
    assert workflow["y"] == 1
    assert workflow["z"] == 2
    assert workflow["u"] == 1


def test_invalid_setvar(evaluator: Any) -> None:
    workflow_config: Dict[str, Any] = {
        "steps": [
            {
                "run": "setvar",
                "with": "1",
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    with pytest.raises(WorkflowError, match=re.escape("Invalid expression: 1")):
        Workflow(parsed_config, context)

    workflow_config = {
        "steps": [
            {
                "run": "setvar",
                "with": "2a = 1",
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    with pytest.raises(WorkflowError, match=re.escape("Invalid identifier: 2a")):
        Workflow(parsed_config, context)


def test_invalid_identifier(evaluator: Any) -> None:
    workflow_config: Dict[str, Any] = {
        "steps": [
            {"run": "setvar", "with": "x=1"},
            {
                "run": "setvar",
                "with": "y=x + 1",
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    with pytest.raises(
        WorkflowError, match=re.escape("Syntax error in workflow expression: x + 1")
    ):
        workflow.run()


def test_conditional_run(enopt_config: EnOptConfig, evaluator: Any) -> None:
    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimal1",
                "init": "tracker",
            },
            {
                "id": "optimal2",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["optimal1"],
                },
                "if": "${{ 1 > 0 }}",
            },
            {"run": "setvar", "with": "x = 1"},
            {
                "run": "optimizer",
                "if": "$x < 0",
                "with": {
                    "config": "$config",
                    "update": ["optimal2"],
                },
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()
    result1 = workflow["optimal1"]
    result2 = workflow["optimal2"]
    assert result1 is not None
    assert np.allclose(result1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert result2 is None


def test_set_initial_values(enopt_config: EnOptConfig, evaluator: Any) -> None:
    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimal1",
                "init": "tracker",
            },
            {
                "id": "optimal2",
                "init": "tracker",
            },
            {
                "id": "optimal3",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["optimal1"],
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["optimal2"],
                    "initial_values": "$optimal1",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["optimal3"],
                    "initial_values": [0, 0, 0],
                },
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()

    result1 = workflow["optimal1"]
    assert result1 is not None
    assert np.allclose(result1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    result2 = workflow["optimal2"]
    assert result2 is not None
    assert np.allclose(result2.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    result3 = workflow["optimal2"]
    assert result3 is not None
    assert np.allclose(result3.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    assert not np.all(result1.evaluations.variables == result2.evaluations.variables)
    assert not np.all(result1.evaluations.variables == result3.evaluations.variables)


def test_reset_results(enopt_config: EnOptConfig, evaluator: Any) -> None:
    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimal",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["optimal"],
                },
            },
            {
                "run": "setvar",
                "with": "saved_results = $optimal",
            },
            {
                "run": "reset",
                "with": "optimal",
            },
        ],
    }
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()

    assert workflow["optimal"] is None
    saved_results = workflow["saved_results"]
    assert saved_results is not None
    assert np.allclose(saved_results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_two_optimizers_alternating(enopt_config: Any, evaluator: Any) -> None:
    completed_functions = 0

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed_functions
        assert event.results is not None
        for item in event.results:
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
    enopt_config1["variables"]["indices"] = [0, 2]
    enopt_config1["optimizer"] = opt_config1
    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["variables"]["indices"] = [1]
    enopt_config2["optimizer"] = opt_config2

    workflow_config = {
        "context": [
            {
                "id": "enopt_config1",
                "init": "config",
                "with": {
                    "config": enopt_config1,
                },
            },
            {
                "id": "enopt_config2",
                "init": "config",
                "with": enopt_config2,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
            {
                "id": "last",
                "init": "tracker",
                "with": {
                    "type": "last",
                },
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                    "update": ["last", "optimum"],
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "update": ["last", "optimum"],
                    "initial_values": "$last",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                    "update": ["last", "optimum"],
                    "initial_values": "$last",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "update": ["optimum"],
                    "initial_values": "$last",
                },
            },
        ],
    }

    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()

    assert completed_functions == 14
    assert workflow["optimum"] is not None
    assert np.allclose(
        workflow["optimum"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optimization_sequential(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 2

    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["optimizer"]["max_functions"] = 3

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "enopt_config2",
                "init": "config",
                "with": enopt_config2,
            },
            {
                "id": "last",
                "init": "tracker",
                "with": {
                    "type": "last",
                },
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "update": ["last"],
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "initial_values": "$last",
                },
            },
        ],
    }

    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()

    assert not np.allclose(
        completed[1].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert np.all(
        completed[2].evaluations.variables == completed[1].evaluations.variables
    )
    assert np.allclose(completed[-1].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_repeat_step(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "update": ["optimum"],
                },
            },
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.run()
    assert workflow["optimum"] is not None
    variables = workflow["optimum"].evaluations.variables.copy()
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

    workflow_config["steps"] = [
        {
            "run": "repeat",
            "with": {
                "iterations": 1,
                "steps": [
                    {
                        "run": "optimizer",
                        "with": {
                            "config": "$enopt_config",
                            "update": ["optimum"],
                        },
                    },
                ],
            },
        }
    ]
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.run()
    assert workflow["optimum"] is not None

    assert np.all(variables == workflow["optimum"].evaluations.variables)

    assert np.all(
        BasicOptimizationWorkflow(enopt_config, evaluator()).repeat(1).run().variables
        == variables
    )


def test_restart_initial(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "config",
                "with": enopt_config,
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 2,
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                            },
                        },
                    ],
                },
            }
        ],
    }

    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()

    assert len(completed) == 6

    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)

    completed = []
    BasicOptimizationWorkflow(enopt_config, evaluator()).add_callback(
        EventType.FINISHED_EVALUATION, _track_evaluations
    ).repeat(2, restart_from="initial").run()
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)


def test_restart_last(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "last",
                "init": "tracker",
                "with": {"type": "last"},
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 2,
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                                "update": ["last"],
                                "initial_values": "$last",
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()

    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )

    completed = []
    BasicOptimizationWorkflow(enopt_config, evaluator()).repeat(
        2, restart_from="last"
    ).add_callback(EventType.FINISHED_EVALUATION, _track_evaluations).run()
    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )


def test_restart_optimum(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 2,
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                                "update": ["optimum"],
                                "initial_values": "$optimum",
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()

    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )

    completed = []
    BasicOptimizationWorkflow(enopt_config, evaluator()).add_callback(
        EventType.FINISHED_EVALUATION, _track_evaluations
    ).repeat(2, restart_from="optimal").run()
    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )


def test_restart_optimum_with_reset(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    completed: List[FunctionResults] = []
    max_functions = 5

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed
        assert event.results is not None
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

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 3,
                    "steps": [
                        {
                            "run": "setvar",
                            "with": "initial = $optimum",
                        },
                        {
                            "run": "reset",
                            "with": {
                                "context": "optimum",
                            },
                        },
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                                "update": ["optimum"],
                                "initial_values": "$initial",
                            },
                        },
                    ],
                },
            },
        ],
    }
    context = OptimizerContext(evaluator=evaluator(new_functions))
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()

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

    completed = []
    BasicOptimizationWorkflow(enopt_config, evaluator(new_functions)).add_callback(
        EventType.FINISHED_EVALUATION, _track_evaluations
    ).repeat(3, restart_from="last_optimal").run()

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


def test_repeat_metadata(enopt_config: EnOptConfig, evaluator: Any) -> None:
    restarts: List[int] = []

    def _track_results(event: OptimizationEvent) -> None:
        assert event.results is not None
        metadata = event.results[0].metadata
        restart = metadata.get("restart", -1)
        assert metadata["foo"] == 1
        assert metadata["bar"] == "string"
        assert metadata["complex"] == f"string 2 {restart}"
        if not restarts or restart != restarts[-1]:
            restarts.append(restart)

    metadata = {
        "restart": "$counter",
        "foo": 1,
        "bar": "string",
        "complex": "string ${{ 1 + 1}} $counter",
    }

    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 2,
                    "counter_var": "counter",
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$config",
                                "metadata": metadata,
                            },
                        },
                    ],
                },
            }
        ],
    }

    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_results
    )
    workflow.run()
    assert restarts == [0, 1]

    restarts = []
    BasicOptimizationWorkflow(enopt_config, evaluator()).add_callback(
        EventType.FINISHED_EVALUATION, _track_results
    ).add_metadata(metadata).repeat(
        2, restart_from="last_optimal", counter_var="counter"
    ).run()
    assert restarts == [0, 1]


def test_update_enopt(enopt_config: Any, evaluator: Any) -> None:
    weights = enopt_config["objective_functions"]["weights"]
    enopt_config["objective_functions"]["weights"] = [1, 1]

    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["optimum"],
                },
            },
        ],
    }

    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()

    assert workflow["optimum"] is not None
    assert not np.allclose(
        workflow["optimum"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )

    workflow_config["steps"] = [
        {
            "run": "update",
            "with": {
                "context": "config",
                "value": {"objective_functions": {"weights": weights}},
            },
        },
    ] + workflow_config["steps"]
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()
    assert workflow["optimum"] is not None
    assert np.allclose(
        workflow["optimum"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_evaluator_step(enopt_config: Any, evaluator: Any) -> None:
    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "evaluator",
                "with": {
                    "config": "$config",
                    "update": ["optimum"],
                },
            },
        ],
    }

    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()

    assert workflow["optimum"] is not None
    assert workflow["optimum"].functions is not None
    assert np.allclose(workflow["optimum"].functions.weighted_objective, 1.66)

    workflow_config["steps"][0]["with"]["values"] = [0, 0, 0]
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()

    assert workflow["optimum"] is not None
    assert workflow["optimum"].functions is not None
    assert np.allclose(workflow["optimum"].functions.weighted_objective, 1.75)


def test_evaluator_step_multi(enopt_config: Any, evaluator: Any) -> None:
    completed: List[float] = []

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item.functions.weighted_objective.item()
            for item in event.results
            if isinstance(item, FunctionResults) and item.functions is not None
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "evaluator",
                "with": {
                    "config": "$config",
                    "update": ["optimum"],
                    "values": [[0, 0, 0.1], [0, 0, 0]],
                },
            },
        ],
    }

    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()

    assert len(completed) == 2
    assert np.allclose(completed, [1.66, 1.75])


def test_nested_workflow(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    completed_functions = 0

    def _track_evaluations(event: OptimizationEvent) -> None:
        nonlocal completed_functions
        assert event.results is not None
        for item in event.results:
            if isinstance(item, FunctionResults):
                completed_functions += 1

    enopt_config["optimizer"]["tolerance"] = 1e-10
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4
    enopt_config["variables"]["indices"] = [0, 2]
    nested_config = deepcopy(enopt_config)
    nested_config["variables"]["indices"] = [1]
    enopt_config["optimizer"]["max_functions"] = 5

    inner_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": nested_config,
            },
            {
                "id": "nested_optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["nested_optimum"],
                    "initial_values": "$initial",
                },
            },
        ],
    }

    outer_config = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": ["optimum"],
                    "nested_workflow": {
                        "workflow": inner_config,
                        "initial_var": "initial",
                        "results_var": "nested_optimum",
                    },
                },
            },
        ],
    }

    parsed_config = WorkflowConfig.model_validate(outer_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_EVALUATION, _track_evaluations
    )
    workflow.run()
    results = workflow["optimum"]

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert completed_functions == 25


def test_exit_code(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    is_called = False

    def _exit_code(
        event: OptimizationEvent,
    ) -> None:
        nonlocal is_called
        is_called = True
        assert isinstance(event, OptimizationEvent)
        assert event.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "config",
                "with": enopt_config,
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "exit_code_var": "exit_code",
                },
            },
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.optimizer_context.events.add_observer(
        EventType.FINISHED_OPTIMIZER_STEP, _exit_code
    )
    workflow.run()
    assert workflow["exit_code"] == OptimizerExitCode.MAX_FUNCTIONS_REACHED
    assert is_called
