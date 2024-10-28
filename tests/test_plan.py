from __future__ import annotations

import filecmp
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
import pytest

from ropt.config.plan import PlanConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.plan import (
    Event,
    ExpressionEvaluator,
    OptimizationPlanRunner,
    OptimizerContext,
    Plan,
)
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
            "results": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {"config": "$enopt_config", "tags": "opt"},
            },
        ],
        "results": [
            {
                "run": "tracker",
                "with": {"var": "results", "tags": "opt"},
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


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("", ""),
        (" ", ""),
        ("${{1}}", 1),
        ("${{'1'}}", "1"),
        ("${{'${{1}}'}}", "${{1}}"),
        (" ${{-1}} ", -1),
        ("${{not 1}}", False),
        ("${{True and False}}", False),
        ("${{True or False}}", True),
        ("${{1 + 1}}", 2),
        ("${{2**3}}", 8),
        ("${{3 % 2}}", 1),
        ("${{3 // 2}}", 1),
        ("${{2.5 + (2 + 3) / 2}}", 5),
        ("${{1 < 2}}", True),
        ("${{1 < 23}}", True),
        ("${{1 < 2 > 3}}", False),
        ("${{$x}}", 1),
        ("${{'$x'}}", "x"),
        ("${{'$$x'}}", "$x"),
        ("${{$x + $y}}", 2),
        ("$x + 1", 2),
        ("1 + $x", "1 + $x"),
        ("${{1 + $y}}", 2),
        ("${{ {'a': {'b': 1}} }}", {"a": {"b": 1}}),
        ("$dummy", None),
        ("$$dummy", "$dummy"),
        ("${{[1, 2]}}", [1, 2]),
        ("${{[$dummy, 2]}}", [None, 2]),
        ("${{max (1, 2)}}", 2),
        ("${{max ($x, 2)}}", 2),
        ("${{$max ($x, 2)}}", 2),
        ("$max($x, 2)", 2),
        ("$incr($x)", 2),
        ("${{$incr($x)}}", 2),
        ("$[[1]]", "1"),
        ("$[[[1]]]", "[1]"),
        ("$[[${{ $x }}]]", "1"),
        ("$[[ ${{not 1}} ]]", " False "),
        ("$[[${{True and False}}]]", "False"),
        ("$[[a ${{ 1 }} b]]", "a 1 b"),
        ("$[[a ${{ 1 + 1 }} b]]", "a 2 b"),
        ("$[[${{ {'a': {'b': 1}} }}]]", "{'a': {'b': 1}}"),
        (
            "$[[${{ {'a': {$i: 1}} }} - ${{ {'a': {'b': $x}} }}]]",
            "{'a': {'b': 1}} - {'a': {'b': 1}}",
        ),
    ],
)
def test_eval_expr(evaluator: Any, expr: str, expected: Any) -> None:
    plan_config: Dict[str, Any] = {
        "variables": {
            "dummy": None,
            "x": 1,
            "y": 1,
            "i": "b",
        },
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(
        evaluator=evaluator(), expr=ExpressionEvaluator({"incr": lambda x: x + 1})
    )
    plan = Plan(parsed_config, context)
    plan.run()

    assert plan.eval(expr) == expected


@pytest.mark.parametrize(
    ("expr", "message", "exception"),
    [
        ("${{1 + * 1}}", "invalid syntax", SyntaxError),
        ("$z", "Unknown plan variable: `z`", AttributeError),
        ("$[[a ${{ 1 + 1 ${{ 1 }} }} b]]", "unmatched '}'", SyntaxError),
        ("$[[${{ 1 + * 1 }}]]", "invalid syntax", SyntaxError),
        ("${{ foo() }}", "Unknown plan function: `foo`", AttributeError),
        ("${{ $foo() }}", "Unknown plan function: `foo`", AttributeError),
        ("${{ incr(1) }}", "Unknown plan function: `incr`", AttributeError),
    ],
)
def test_eval_exception(
    evaluator: Any,
    expr: str,
    message: Any,
    exception: Union[AttributeError, SyntaxError],
) -> None:
    plan_config: Dict[str, Any] = {}
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(
        evaluator=evaluator(), expr=ExpressionEvaluator({"incr": lambda x: x + 1})
    )
    plan = Plan(parsed_config, context)
    plan.run()
    with pytest.raises(exception, match=message):  # type: ignore[call-overload]
        plan.eval(expr)


def test_eval_attribute(enopt_config: Any, evaluator: Any) -> None:
    plan_config: Dict[str, Any] = {
        "variables": {
            "config": enopt_config,
            "dummy": None,
            "results": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": "opt",
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "results", "tags": "opt"}},
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()
    assert isinstance(plan.eval("$results"), Results)
    assert plan.eval("$results.result_id") >= 0


@pytest.mark.parametrize(
    ("variables", "expr", "check", "expected"),
    [
        ({"x": None}, {"x": 1}, "x", 1),
        ({"x": None, "y": None}, {"x": 1, "y": "${{$x + 1}}"}, "y", 2),
        ({"x": {"a": 1}, "i": "a"}, {"x[$i]": 2}, "x", {"a": 2}),
        ({"x": {"a": {10: {}}}}, {"x['a'][10]": 1}, "x", {"a": {10: 1}}),
    ],
)
def test_set_step(
    evaluator: Any,
    variables: Dict[str, Any],
    expr: Dict[str, Any],
    check: str,
    expected: Any,
) -> None:
    plan_config: Dict[str, Any] = {
        "variables": variables,
        "steps": [{"set": expr}],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()
    assert plan[check] == expected


@pytest.mark.parametrize(
    ("variables", "expr", "check", "expected"),
    [
        (
            {"o": type("Obj", (object,), {"b_b": type("Obj", (object,), {"c1": 1})})},
            {"o.b_b.c1": 2},
            'plan["o"].b_b.c1',
            2,
        ),
        (
            {
                "o": type(
                    "Obj", (object,), {"b_b": type("Obj", (object,), {"c1": {20: 1}})}
                )
            },
            {"o.b_b.c1[20]": 2},
            'plan["o"].b_b.c1',
            {20: 2},
        ),
        (
            {"o": type("Obj", (object,), {"b_b": [type("Obj", (object,), {"c1": 1})]})},
            {"o.b_b[0].c1": 2},
            'plan["o"].b_b[0].c1',
            2,
        ),
    ],
)
def test_set_step_attribute(
    evaluator: Any,
    variables: Dict[str, Any],
    expr: Dict[str, Any],
    check: str,
    expected: Any,
) -> None:
    plan_config: Dict[str, Any] = {
        "variables": variables,
        "steps": [{"set": expr}],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.run()
    assert eval(check) == expected  # noqa: S307


def test_set_keys_exception(evaluator: Any) -> None:
    plan_config: Dict[str, Any] = {
        "variables": {
            "y": {"a": None},
        },
        "steps": [
            {"set": {"y['a']['b']": 1}},
        ],
    }
    parsed_config = PlanConfig.model_validate(plan_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    with pytest.raises(
        AttributeError, match=re.escape("Invalid attribute access: y['a']['b']")
    ):
        plan.run()


def test_conditional_run(enopt_config: EnOptConfig, evaluator: Any) -> None:
    plan_config = {
        "variables": {
            "config": enopt_config,
            "optimal1": None,
            "optimal2": None,
            "x": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": "optimal1",
                },
                "if": "${{1 > 0}}",
            },
            {"set": {"x": 1}},
            {
                "run": "optimizer",
                "if": "$x < 0",
                "with": {
                    "config": "$config",
                    "tags": "optimal2",
                },
            },
        ],
        "results": [
            {
                "run": "tracker",
                "with": {"var": "optimal1", "tags": "optimal1"},
            },
            {
                "run": "tracker",
                "with": {"var": "optimal2", "tags": "optimal2"},
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
            "optimal1": None,
            "optimal2": None,
            "optimal3": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": "optimal1",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": "optimal2",
                    "initial_values": "$optimal1",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": "optimal3",
                    "initial_values": [0, 0, 0],
                },
            },
        ],
        "results": [
            {
                "run": "tracker",
                "with": {"var": "optimal1", "tags": ["optimal1"]},
            },
            {
                "run": "tracker",
                "with": {"var": "optimal2", "tags": ["optimal2"]},
            },
            {
                "run": "tracker",
                "with": {"var": "optimal3", "tags": ["optimal3"]},
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
            "optimal": None,
            "saved_results": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "tags": "opt",
                },
            },
            {
                "set": {
                    "saved_results": "$optimal",
                    "optimal": None,
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "optimal", "tags": "opt"}},
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
            "optimum": None,
            "last": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                    "tags": "opt",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "initial_values": "$last",
                    "tags": "opt",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config1",
                    "initial_values": "$last",
                    "tags": "opt",
                },
            },
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config2",
                    "initial_values": "$last",
                    "tags": "opt",
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "optimum", "tags": "opt"}},
            {
                "run": "tracker",
                "with": {"var": "last", "type": "last", "tags": "opt"},
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
            "last": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "tags": "last",
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
        "results": [
            {
                "run": "tracker",
                "with": {"var": "last", "type": "last", "tags": "last"},
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
            "optimum": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$enopt_config",
                    "tags": "opt",
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "optimum", "tags": "opt"}},
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
                            "tags": "opt",
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
            "last": None,
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
                                "initial_values": "$last",
                                "tags": "opt",
                            },
                        },
                    ],
                },
            }
        ],
        "results": [
            {
                "run": "tracker",
                "with": {"var": "last", "type": "last", "tags": "opt"},
            },
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
            "optimum": None,
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
                                "tags": "opt",
                                "initial_values": "$optimum",
                            },
                        },
                    ],
                },
            }
        ],
        "results": [
            {"run": "tracker", "with": {"var": "optimum", "tags": "opt"}},
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
            "optimum": None,
            "initial": None,
        },
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 3,
                    "steps": [
                        {
                            "set": {
                                "initial": "$optimum",
                                "optimum": None,
                            },
                        },
                        {
                            "run": "optimizer",
                            "with": {
                                "config": "$enopt_config",
                                "initial_values": "$initial",
                                "tags": "opt",
                            },
                        },
                    ],
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "optimum", "tags": "opt"}},
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
        "complex": "$[[string ${{ 1 + 1 }} ${{ $counter }}]]",
    }

    plan_config = {
        "variables": {
            "config": enopt_config,
            "metadata": metadata,
            "counter": 0,
        },
        "steps": [
            {
                "run": "repeat",
                "with": {
                    "iterations": 2,
                    "var": "counter",
                    "steps": [
                        {
                            "run": "optimizer",
                            "with": {"config": "$config", "tags": "opt"},
                        },
                    ],
                },
            }
        ],
        "results": [
            {"run": "metadata", "with": {"data": metadata, "tags": "opt"}},
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
            "optimum": None,
        },
        "steps": [
            {
                "run": "evaluator",
                "with": {
                    "config": "$config",
                    "tags": "opt",
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "optimum", "tags": "opt"}},
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
            "optimum": None,
        },
        "steps": [
            {
                "run": "evaluator",
                "with": {
                    "config": "$config",
                    "values": [[0, 0, 0.1], [0, 0, 0]],
                    "tags": "opt",
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "optimum", "tags": "opt"}},
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
            "exit_code": None,
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
        "inputs": ["initial", "config"],
        "outputs": ["nested_optimum"],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "initial_values": "$initial",
                    "tags": "inner",
                },
            },
        ],
        "results": [
            {
                "run": "tracker",
                "with": {"var": "nested_optimum", "tags": "inner"},
            },
        ],
    }

    outer_config = {
        "variables": {
            "config": enopt_config,
            "optimum": None,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "nested_optimization": {
                        "plan": inner_config,
                        "extra_inputs": [nested_config],
                    },
                    "tags": "outer",
                },
            },
        ],
        "results": [
            {
                "run": "tracker",
                "with": {"var": "optimum", "tags": "outer"},
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


def test_nested_plan_metadata(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.2, 0.1]

    def _track_evaluations(event: Event) -> None:
        assert event.results is not None
        for item in event.results:
            if isinstance(item, FunctionResults):
                assert item.metadata.get("outer") == 1
                if "inner" in event.tags:
                    assert item.metadata.get("inner") == "inner_meta_data"

    enopt_config["optimizer"]["tolerance"] = 1e-10
    enopt_config["optimizer"]["speculative"] = True
    enopt_config["optimizer"]["max_functions"] = 4
    enopt_config["variables"]["indices"] = [0, 2]
    nested_config = deepcopy(enopt_config)
    nested_config["variables"]["indices"] = [1]
    enopt_config["optimizer"]["max_functions"] = 5

    inner_config = {
        "inputs": ["initial"],
        "outputs": ["nested_optimum"],
        "variables": {
            "config": nested_config,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "initial_values": "$initial",
                    "tags": "inner",
                },
            },
        ],
        "results": [
            {"run": "tracker", "with": {"var": "nested_optimum", "tags": "inner"}},
            {
                "run": "metadata",
                "with": {"data": {"inner": "inner_meta_data"}, "tags": "inner"},
            },
        ],
    }

    outer_config = {
        "variables": {
            "config": enopt_config,
            "optimum": None,
            "x": 1,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "nested_optimization": {"plan": inner_config},
                    "tags": "outer",
                },
            },
        ],
        "results": [
            {
                "run": "metadata",
                "with": {"data": {"outer": "$x"}, "tags": ["inner", "outer"]},
            },
            {"run": "tracker", "with": {"var": "optimum", "tags": "inner"}},
        ],
    }

    parsed_config = PlanConfig.model_validate(outer_config)
    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(parsed_config, context)
    plan.add_observer(EventType.FINISHED_EVALUATION, _track_evaluations)
    plan.run()
    results = plan["optimum"]

    assert results.metadata["inner"] == "inner_meta_data"
    assert results.metadata["outer"] == 1

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


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


def test_table_handler(enopt_config: Any, evaluator: Any, tmp_path: Path) -> None:
    enopt_config["optimizer"]["max_functions"] = 5

    path1 = tmp_path / "results1.txt"
    plan_config = {
        "variables": {
            "enopt_config": enopt_config,
        },
        "steps": [
            {
                "run": "optimizer",
                "with": {"config": "$enopt_config", "tags": "opt"},
            },
        ],
        "results": [
            {
                "run": "table",
                "with": {
                    "tags": "opt",
                    "columns": {
                        "result_id": "eval-ID",
                        "evaluations.variables": "Variables",
                    },
                    "path": path1,
                },
            },
        ],
    }

    context = OptimizerContext(evaluator=evaluator())
    plan = Plan(PlanConfig.model_validate(plan_config), context)
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
