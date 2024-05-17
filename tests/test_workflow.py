from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.config.enopt import EnOptConfig
from ropt.config.workflow import WorkflowConfig
from ropt.exceptions import WorkflowError
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicWorkflow, OptimizerContext, Workflow

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
    variables = BasicWorkflow(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_invalid_context_ids() -> None:
    workflow_config: Dict[str, Any] = {
        "context": [
            {
                "id": "1optimal",
                "init": "results_tracker",
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
                "init": "results_tracker",
            },
            {
                "id": "optimal",
                "init": "results_tracker",
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
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "results",
                "init": "results_tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["results"],
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
    assert workflow.parse_value("${{ [results, 2] }}") == [None, 2]

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
        match=re.escape("Unknown workflow variable or context object: `y`"),
    ):
        workflow.parse_value("${{ y + 1 }}")

    workflow.run()

    assert isinstance(workflow.parse_value("$results"), Results)


def test_conditional_run(enopt_config: EnOptConfig, evaluator: Any) -> None:
    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimal1",
                "init": "results_tracker",
            },
            {
                "id": "optimal2",
                "init": "results_tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["optimal1", "last_update"],
                },
                "if": "${{ 1 > 0 }}",
            },
            {
                "run": "optimizer",
                "if": "1 < 0",
                "with": {
                    "config": "$config",
                    "update_results": ["optimal2"],
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
    assert isinstance(workflow["last_update"], tuple)
    assert isinstance(workflow["last_update"][0], Results)


def test_set_initial_values(enopt_config: EnOptConfig, evaluator: Any) -> None:
    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimal1",
                "init": "results_tracker",
            },
            {
                "id": "optimal2",
                "init": "results_tracker",
            },
            {
                "id": "optimal3",
                "init": "results_tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["optimal1"],
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["optimal2"],
                    "initial_variables": "$optimal1",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["optimal3"],
                    "initial_variables": [0, 0, 0],
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
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimal",
                "init": "results_tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["optimal"],
                },
            },
            {
                "run": "reset_context",
                "with": {
                    "context_id": "optimal",
                    "backup_id": "saved_results",
                },
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

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed_functions
        for item in results:
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
                "init": "enopt_config",
                "with": enopt_config1,
            },
            {
                "id": "enopt_config2",
                "init": "enopt_config",
                "with": enopt_config2,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
            {
                "id": "last",
                "init": "results_tracker",
                "with": {
                    "type": "last",
                },
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                    "update_results": ["last", "optimum", "callback"],
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "update_results": ["last", "optimum", "callback"],
                    "initial_variables": "$last",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                    "update_results": ["last", "optimum", "callback"],
                    "initial_variables": "$last",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "update_results": ["optimum", "callback"],
                    "initial_variables": "$last",
                },
            },
        ],
    }

    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.run()

    assert completed_functions == 14
    assert workflow["optimum"] is not None
    assert np.allclose(
        workflow["optimum"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optimization_sequential(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed
        completed += [item for item in results if isinstance(item, FunctionResults)]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 2

    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["optimizer"]["max_functions"] = 3

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "enopt_config2",
                "init": "enopt_config",
                "with": enopt_config2,
            },
            {
                "id": "last",
                "init": "results_tracker",
                "with": {
                    "type": "last",
                },
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "update_results": ["last", "callback"],
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "initial_variables": "$last",
                    "update_results": ["callback"],
                },
            },
        ],
    }

    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
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
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "update_results": ["optimum"],
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
                            "update_results": ["optimum"],
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


def test_restart_initial(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed
        completed += [item for item in results if isinstance(item, FunctionResults)]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
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
                                "update_results": ["callback"],
                            },
                        },
                    ],
                },
            }
        ],
    }

    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.run()

    assert len(completed) == 6

    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)


def test_restart_last(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed
        completed += [item for item in results if isinstance(item, FunctionResults)]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "last",
                "init": "results_tracker",
                "with": {"type": "last"},
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
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
                                "update_results": ["last", "callback"],
                                "initial_variables": "$last",
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.run()

    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )


def test_restart_optimum(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed
        completed += [item for item in results if isinstance(item, FunctionResults)]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
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
                                "update_results": ["optimum", "callback"],
                                "initial_variables": "$optimum",
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.run()

    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )


def test_restart_optimum_with_reset(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    completed: List[FunctionResults] = []
    max_functions = 5

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed
        completed += [item for item in results if isinstance(item, FunctionResults)]

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
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 3,
                    "steps": [
                        {
                            "run": "reset_context",
                            "with": {
                                "context_id": "optimum",
                                "backup_id": "initial",
                            },
                        },
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                                "update_results": ["optimum", "callback"],
                                "initial_variables": "$initial",
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator(new_functions))
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
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


def test_repeat_metadata(enopt_config: EnOptConfig, evaluator: Any) -> None:
    restarts: List[int] = []

    def track_results(results: Tuple[Results, ...]) -> None:
        metadata = results[0].metadata
        restart = metadata.get("restart", -1)
        assert metadata["foo"] == 1
        assert metadata["bar"] == "string"
        assert metadata["complex"] == f"string 2 {restart}"
        if not restarts or restart != restarts[-1]:
            restarts.append(restart)

    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": track_results},
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 2,
                    "counter_id": "counter",
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$config",
                                "update_results": ["callback"],
                                "metadata": {
                                    "restart": "$counter",
                                    "foo": 1,
                                    "bar": "string",
                                    "complex": "string ${{ 1 + 1}} $counter",
                                },
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
    workflow.run()
    assert restarts == [0, 1]


def test_update_enopt(enopt_config: Any, evaluator: Any) -> None:
    weights = enopt_config["objective_functions"]["weights"]
    enopt_config["objective_functions"]["weights"] = [1, 1]

    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["optimum"],
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
            "run": "update_context",
            "with": {
                "context_id": "config",
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
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
        ],
        "steps": [
            {
                "run": "evaluator",
                "with": {
                    "config": "$config",
                    "update_results": ["optimum"],
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

    workflow_config["steps"][0]["with"]["variables"] = [0, 0, 0]
    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()

    assert workflow["optimum"] is not None
    assert workflow["optimum"].functions is not None
    assert np.allclose(workflow["optimum"].functions.weighted_objective, 1.75)


def test_evaluator_step_multi(enopt_config: Any, evaluator: Any) -> None:
    completed: List[float] = []

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed
        completed += [
            item.functions.weighted_objective.item()
            for item in results
            if isinstance(item, FunctionResults) and item.functions is not None
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    workflow_config = {
        "context": [
            {
                "id": "config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
            },
        ],
        "steps": [
            {
                "run": "evaluator",
                "with": {
                    "config": "$config",
                    "update_results": ["optimum", "callback"],
                    "variables": [[0, 0, 0.1], [0, 0, 0]],
                },
            },
        ],
    }

    parsed_config = WorkflowConfig.model_validate(workflow_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()

    assert len(completed) == 2
    assert np.allclose(completed, [1.66, 1.75])


def test_nested_workflow(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    completed_functions = 0

    def _track_evaluations(results: Tuple[Results, ...]) -> None:
        nonlocal completed_functions
        for item in results:
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
                "init": "enopt_config",
                "with": nested_config,
            },
            {
                "id": "nested_optimum",
                "init": "results_tracker",
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["nested_optimum", "callback"],
                    "initial_variables": "$initial",
                },
            },
        ],
    }

    outer_config = {
        "context": [
            {
                "id": "config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "optimum",
                "init": "results_tracker",
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {"callback": _track_evaluations},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update_results": ["optimum", "callback"],
                    "nested_workflow": {
                        "workflow": inner_config,
                        "initial_variables_id": "initial",
                        "results_id": "nested_optimum",
                    },
                },
            },
        ],
    }

    parsed_config = WorkflowConfig.model_validate(outer_config)
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(parsed_config, context)
    workflow.run()
    results = workflow["optimum"]

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert completed_functions == 25


def test_callback(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    counts: List[int] = []

    def _store_value(_: Tuple[Results, ...], count: int, config: EnOptConfig) -> int:
        assert isinstance(config, EnOptConfig)
        if not counts or count != counts[-1]:
            counts.append(count)
        return count

    workflow_config = {
        "context": [
            {
                "id": "enopt_config",
                "init": "enopt_config",
                "with": enopt_config,
            },
            {
                "id": "callback",
                "init": "results_callback",
                "with": {
                    "callback": _store_value,
                    "kwargs": {
                        "count": "$counter",
                        "config": "$enopt_config",
                    },
                },
            },
        ],
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 3,
                    "counter_id": "counter",
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                                "update_results": ["callback"],
                                "metadata": {
                                    "restart": "$counter",
                                },
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    workflow = Workflow(WorkflowConfig.model_validate(workflow_config), context)
    workflow.run()
    assert counts == [0, 1, 2]
    assert workflow["callback"] == 2
