from __future__ import annotations

import filecmp
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.config.plan import PlanConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import PlanError
from ropt.optimization import Event
from ropt.plan import OptimizationPlanRunner, OptimizerContext, Plan
from ropt.report import ResultsTable
from ropt.results import FunctionResults, Results

if TYPE_CHECKING:
    from pathlib import Path

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
    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
        "context": [
            {
                "id": "results",
                "init": "tracker",
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {"config": "$enopt_config"},
            },
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.run()
    variables = plan["results"].evaluations.variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

    variables = OptimizationPlanRunner(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_invalid_context_ids() -> None:
    plan_config: Dict[str, Any] = {
        "context": [
            {
                "id": "1optimal",
                "init": "tracker",
            },
        ],
        "steps": [],
    }
    with pytest.raises(ValidationError, match=".*Invalid ID: 1optimal.*"):
        PlanConfig.model_validate(plan_config)


def test_duplicate_context_ids() -> None:
    plan_config: Dict[str, Any] = {
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
        PlanConfig.model_validate(plan_config)


def test_parse_value(enopt_config: Any, evaluator: Any) -> None:
    plan_config: Dict[str, Any] = {
        "variables": {
            "config": enopt_config,
        },
        "context": [
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
                },
            },
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    assert plan.parse_value("${{ 1 }}") == 1
    assert plan.parse_value("${{ -1 }}") == -1
    assert not plan.parse_value("${{ not 1 }}")
    assert not plan.parse_value("${{ True and False }}")
    assert plan.parse_value("${{ True or False }}")
    assert plan.parse_value("${{ 1 + 1 }}") == 2
    assert plan.parse_value("${{ 2**3 }}") == 8
    assert plan.parse_value("${{ 3 % 2 }}") == 1
    assert plan.parse_value("${{ 3 // 2 }}") == 1
    assert plan.parse_value("${{ 2.5 + (2 + 3) / 2 }}") == 5
    assert plan.parse_value("${{ 1 < 2 }}")
    assert plan.parse_value("${{ 1 < 2 < 3 }}")
    assert not plan.parse_value("${{ 1 < 2 > 3 }}")

    assert plan.parse_value("$results") is None
    assert plan.parse_value("${{ [1, 2] }}") == [1, 2]
    assert plan.parse_value("${{ [$results, 2] }}") == [None, 2]

    assert plan.parse_value("a ${{ 1 }} b") == "a 1 b"
    assert plan.parse_value("a ${{ 1 + 1 }} b") == "a 2 b"
    assert plan.parse_value("a ${{ 1 + 1 }} b $results") == "a 2 b None"
    assert plan.parse_value("a ${{ 1 + 1 }} b $$results") == "a 2 b $results"

    with pytest.raises(
        PlanError,
        match=re.escape("Syntax error in expression: 1 + 1 ${{ x"),
    ):
        plan.parse_value("a $results ${{ 1 + 1 ${{ x }} }} b")

    with pytest.raises(
        PlanError, match=re.escape("Syntax error in expression: 1 + * 1")
    ):
        plan.parse_value("${{ 1 + * 1 }}")

    with pytest.raises(
        PlanError,
        match=re.escape("Unknown plan variable: `y`"),
    ):
        plan.parse_value("${{ $y + 1 }}")

    plan.run()

    assert isinstance(plan.parse_value("$results"), Results)


def test_setvar(evaluator: Any) -> None:
    plan_config: Dict[str, Any] = {
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
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()
    assert plan["x"] == 1
    assert plan["y"] == 1
    assert plan["z"] == 2
    assert plan["u"] == 1


def test_invalid_setvar(evaluator: Any) -> None:
    plan_config: Dict[str, Any] = {
        "steps": [
            {
                "run": "setvar",
                "with": "1",
            },
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    with pytest.raises(PlanError, match=re.escape("Invalid expression: 1")):
        Plan(parsed_config, context)

    plan_config = {
        "steps": [
            {
                "run": "setvar",
                "with": "2a = 1",
            },
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    with pytest.raises(PlanError, match=re.escape("Invalid identifier: 2a")):
        Plan(parsed_config, context)


def test_invalid_identifier(evaluator: Any) -> None:
    plan_config: Dict[str, Any] = {
        "steps": [
            {"run": "setvar", "with": "x=1"},
            {
                "run": "setvar",
                "with": "y=x + 1",
            },
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    with pytest.raises(PlanError, match=re.escape("Syntax error in expression: x + 1")):
        plan.run()


def test_conditional_run(enopt_config: EnOptConfig, evaluator: Any) -> None:
    plan_config = {
        "variables": {
            "config": enopt_config,
        },
        "context": [
            {
                "id": "optimal1",
                "init": "tracker",
                "with": {"filter": ["optimal1"]},
            },
            {
                "id": "optimal2",
                "init": "tracker",
                "with": {"filter": ["optimal2"]},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": ["optimal1"],
                },
                "if": "${{ 1 > 0 }}",
            },
            {"run": "setvar", "with": "x = 1"},
            {
                "run": "optimizer",
                "if": "$x < 0",
                "with": {
                    "config": "$config",
                    "tags": ["optimal2"],
                },
            },
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()
    result1 = plan["optimal1"]
    result2 = plan["optimal2"]
    assert result1 is not None
    assert np.allclose(result1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert result2 is None


def test_set_initial_values(enopt_config: EnOptConfig, evaluator: Any) -> None:
    plan_config = {
        "variables": {
            "config": enopt_config,
        },
        "context": [
            {
                "id": "optimal1",
                "init": "tracker",
                "with": {"filter": ["optimal1"]},
            },
            {
                "id": "optimal2",
                "init": "tracker",
                "with": {"filter": ["optimal2"]},
            },
            {
                "id": "optimal3",
                "init": "tracker",
                "with": {"filter": ["optimal3"]},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": ["optimal1"],
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": ["optimal2"],
                    "initial_values": "$optimal1",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": ["optimal3"],
                    "initial_values": [0, 0, 0],
                },
            },
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()

    result1 = plan["optimal1"]
    assert result1 is not None
    assert np.allclose(result1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    result2 = plan["optimal2"]
    assert result2 is not None
    assert np.allclose(result2.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    result3 = plan["optimal2"]
    assert result3 is not None
    assert np.allclose(result3.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    assert not np.all(result1.evaluations.variables == result2.evaluations.variables)
    assert not np.all(result1.evaluations.variables == result3.evaluations.variables)


def test_reset_results(enopt_config: EnOptConfig, evaluator: Any) -> None:
    plan_config = {
        "variables": {
            "config": enopt_config,
        },
        "context": [
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
                },
            },
            {
                "run": "setvar",
                "with": "saved_results = $optimal",
            },
            {
                "run": "setvar",
                "with": "optimal = None",
            },
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()

    assert plan["optimal"] is None
    saved_results = plan["saved_results"]
    assert saved_results is not None
    assert np.allclose(saved_results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_two_optimizers_alternating(enopt_config: Any, evaluator: Any) -> None:
    completed_functions = 0

    def _track_evaluations(event: Event) -> None:
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

    plan_config = {
        "variables": {
            "enopt_config1": enopt_config1,
            "enopt_config2": enopt_config2,
        },
        "context": [
            {
                "id": "optimum",
                "init": "tracker",
            },
            {
                "id": "last",
                "init": "tracker",
                "with": {"type": "last"},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "initial_values": "$last",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                    "initial_values": "$last",
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
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()

    assert completed_functions == 14
    assert plan["optimum"] is not None
    assert np.allclose(
        plan["optimum"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optimization_sequential(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 2

    enopt_config2 = deepcopy(enopt_config)
    enopt_config2["optimizer"]["max_functions"] = 3

    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
            "enopt_config2": enopt_config2,
        },
        "context": [
            {
                "id": "last",
                "init": "tracker",
                "with": {"type": "last", "filter": ["last"]},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "tags": ["last"],
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
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()

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

    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
        "context": [
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
                },
            },
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.run()
    assert plan["optimum"] is not None
    variables = plan["optimum"].evaluations.variables.copy()
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

    plan_config["steps"] = [
        {
            "run": "repeat",
            "with": {
                "iterations": 1,
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
    ]
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.run()
    assert plan["optimum"] is not None

    assert np.all(variables == plan["optimum"].evaluations.variables)


def test_restart_initial(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
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
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()

    assert len(completed) == 6

    initial = np.array([enopt_config["variables"]["initial_values"]])
    assert np.all(completed[0].evaluations.variables == initial)
    assert np.all(completed[3].evaluations.variables == initial)


def test_restart_last(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 3

    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
        "context": [
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
                                "initial_values": "$last",
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()

    assert np.all(
        completed[3].evaluations.variables == completed[2].evaluations.variables
    )


def test_restart_optimum(enopt_config: Any, evaluator: Any) -> None:
    completed: List[FunctionResults] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item for item in event.results if isinstance(item, FunctionResults)
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
        "context": [
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
                                "initial_values": "$optimum",
                            },
                        },
                    ],
                },
            }
        ],
    }
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()

    assert np.all(
        completed[2].evaluations.variables == completed[4].evaluations.variables
    )


def test_restart_optimum_with_reset(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    completed: List[FunctionResults] = []
    max_functions = 5

    def _track_evaluations(event: Event) -> None:
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

    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
        "context": [
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
                            "run": "setvar",
                            "with": "optimum = None",
                        },
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                                "initial_values": "$initial",
                            },
                        },
                    ],
                },
            },
        ],
    }
    context = OptimizerContext(evaluator=evaluator(new_functions))
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()

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

    def _track_results(event: Event) -> None:
        assert event.results is not None
        metadata = event.results[0].metadata
        restart = metadata.get("restart", -1)
        assert metadata["foo"] == 1
        assert metadata["bar"] == "string"
        if "complex" in metadata:
            assert metadata["complex"] == f"string 2 {restart}"
        if not restarts or restart != restarts[-1]:
            restarts.append(restart)

    metadata = {
        "restart": "$counter",
        "foo": 1,
        "bar": "string",
        "complex": "string ${{ 1 + 1}} $counter",
    }

    plan_config = {
        "variables": {
            "config": enopt_config,
            "metadata": metadata,
        },
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 2,
                    "counter_var": "counter",
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {"config": "$config", "metadata": metadata},
                        },
                    ],
                },
            }
        ],
    }

    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_results)
    plan.run()
    assert restarts == [0, 1]


def test_evaluator_step(enopt_config: Any, evaluator: Any) -> None:
    plan_config: Dict[str, Any] = {
        "variables": {
            "config": enopt_config,
        },
        "context": [
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
                },
            },
        ],
    }

    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()

    assert plan["optimum"] is not None
    assert plan["optimum"].functions is not None
    assert np.allclose(plan["optimum"].functions.weighted_objective, 1.66)

    plan_config["steps"][0]["with"]["values"] = [0, 0, 0]
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()

    assert plan["optimum"] is not None
    assert plan["optimum"].functions is not None
    assert np.allclose(plan["optimum"].functions.weighted_objective, 1.75)


def test_evaluator_step_multi(enopt_config: Any, evaluator: Any) -> None:
    completed: List[float] = []

    def _track_evaluations(event: Event) -> None:
        nonlocal completed
        assert event.results is not None
        completed += [
            item.functions.weighted_objective.item()
            for item in event.results
            if isinstance(item, FunctionResults) and item.functions is not None
        ]

    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    plan_config = {
        "variables": {
            "config": enopt_config,
        },
        "context": [
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
                    "values": [[0, 0, 0.1], [0, 0, 0]],
                },
            },
        ],
    }

    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.add_observer(EventType.FINISHED_EVALUATOR_STEP, _track_evaluations)
    plan.run()

    assert len(completed) == 2
    assert np.allclose(completed, [1.66, 1.75])


def test_exit_code(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4

    is_called = False

    def _exit_code(
        event: Event,
    ) -> None:
        nonlocal is_called
        is_called = True
        assert isinstance(event, Event)
        assert event.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED

    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
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
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_OPTIMIZER_STEP, _exit_code)
    plan.run()
    assert plan["exit_code"] == OptimizerExitCode.MAX_FUNCTIONS_REACHED
    assert is_called


def test_nested_plan(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    completed_functions = 0

    def _track_evaluations(event: Event) -> None:
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
        "variables": {
            "config": nested_config,
        },
        "context": [
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
                    "tags": ["nested_optimum"],
                    "initial_values": "$initial",
                },
            },
        ],
        "inputs": ["initial"],
        "outputs": ["nested_optimum"],
    }

    outer_config = {
        "variables": {
            "config": enopt_config,
        },
        "context": [
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
                    "nested_plan": inner_config,
                },
            },
        ],
    }

    parsed_config = PlanConfig.model_validate(outer_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()
    results = plan["optimum"]

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert completed_functions == 25


def test_table(enopt_config: Any, evaluator: Any, tmp_path: Path) -> None:
    enopt_config["optimizer"]["max_functions"] = 5

    path1 = tmp_path / "results1.txt"
    table = ResultsTable(
        columns={
            "result_id": "eval-ID",
            "evaluations.variables": "Variables",
        },
        path=path1,
    )
    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
        "steps": [
            {
                "name": "opt",
                "run": "optimizer",
                "with": {"config": "$enopt_config"},
            },
        ],
    }

    def handle_results(event: Event) -> None:
        assert event.results is not None
        table.add_results(event.config, event.results)

    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    plan.add_observer(EventType.FINISHED_EVALUATION, handle_results)
    plan.run()

    assert path1.exists()
    with path1.open() as fp:
        assert len(fp.readlines()) == 8

    path2 = tmp_path / "results2.txt"
    OptimizationPlanRunner(enopt_config, evaluator()).add_table(
        columns={"result_id": "eval-ID", "evaluations.variables": "Variables"},
        path=path2,
    ).run()
    assert path2.exists()

    assert filecmp.cmp(path1, path2)
