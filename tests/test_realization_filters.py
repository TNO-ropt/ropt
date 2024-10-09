from functools import partial
from typing import Any, Dict, List

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.enums import ConstraintType, EventType
from ropt.plan import Event, OptimizationPlanRunner
from ropt.plugins.realization_filter.default import (
    _get_cvar_weights_from_percentile,
    _sort_and_select,
)
from ropt.results import FunctionResults, GradientResults, Results


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "optimizer": {
            "speculative": True,
            "max_functions": 10,
        },
        "objective_functions": {
            "names": ["f1", "f2"],
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "number_of_perturbations": 5,
            "perturbation_magnitudes": 0.01,
        },
        "realizations": {
            "weights": 5 * [1.0],
        },
        "variables": {
            "initial_values": 3 * [0],
        },
    }


@pytest.mark.parametrize(
    ("failed", "first", "last", "expected"),
    [
        ([False, False, False, False, False], 0, 1, [0, 0, 0, 0.4, 0.5]),
        ([False, False, False, False, False], 1, 2, [0, 0, 0.3, 0.4, 0]),
        ([False, False, False, True, False], 0, 1, [0, 0, 0.3, 0, 0.5]),
        ([False, False, True, False, False], 1, 2, [0.1, 0, 0.0, 0.4, 0]),
    ],
)
def test__sort_and_select(
    failed: List[bool], first: int, last: int, expected: List[float]
) -> None:
    values = np.array([3, 4, 2, 1, 0])
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = _sort_and_select(values, weights, np.array(failed), first, last)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    ("failed", "percentile", "weights"),
    [
        ([False, False, False, False, False], 0.1, [0.0, 0.0, 0.0, 0.0, 0.1]),
        ([False, False, False, False, False], 0.2, [0.0, 0.0, 0.0, 0.0, 0.2]),
        ([False, False, False, False, False], 0.3, [0.0, 0.0, 0.0, 0.1, 0.2]),
        ([False, False, False, False, False], 0.35, [0.0, 0.0, 0.0, 0.15, 0.2]),
        ([False, False, False, False, False], 0.4, [0.0, 0.0, 0.0, 0.2, 0.2]),
        ([False, False, False, False, False], 0.9, [0.2, 0.1, 0.2, 0.2, 0.2]),
        ([False, False, False, False, False], 1.0, [0.2, 0.2, 0.2, 0.2, 0.2]),
        ([False, False, False, True, False], 0.4, [0, 0, 0.15, 0.0, 0.25]),
    ],
)
def test__get_cvar_weights_from_percentile(
    failed: List[bool], percentile: float, weights: List[float]
) -> None:
    values = np.array([3, 4, 2, 1, 0])
    estimated_weights = _get_cvar_weights_from_percentile(
        values,
        np.array(failed),
        percentile,
    )
    assert np.allclose(estimated_weights, weights)


def _objective_function(
    variables: NDArray[np.float64],
    context: Any,
    target: NDArray[np.float64],
) -> float:
    diff: NDArray[np.float64] = variables - target
    # Make sure that some realizations yield a different result:
    if context.realization % 2 == 0:
        diff += 1.0
    result = np.sum(diff**2)
    # Make sure the other realizations have a worse result:
    if context.realization % 2 == 1:
        result += 10.0
    return float(result)


def _constraint_function(variables: NDArray[np.float64], context: Any) -> float:
    # Track how often this function is called.
    result = variables[0] + variables[2]
    # Break some realizations, same as in the distance function:
    if context.realization % 2 == 0:
        result = variables[0] + variables[2] - 10.0
    return float(result)


def _track_results(event: Event, result_list: List[Results]) -> None:
    assert event.results
    result_list.extend(event.results)


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_sort_filter_on_objectives(
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["optimizer"]["split_evaluations"] = split_evaluations

    variables = (
        OptimizationPlanRunner(enopt_config, evaluator(functions)).run().variables
    )
    assert variables is not None
    assert not np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

    enopt_config["realization_filters"] = [
        {
            "method": "sort-objective",
            "options": {
                "sort": ["f1"],
                "first": 3,
                "last": 4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0]

    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(
            enopt_config,
            evaluator(functions),
        )
        .add_observer(
            EventType.FINISHED_EVALUATION,
            partial(_track_results, result_list=result_list),
        )
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_sort_filter_on_objectives_with_constraints(
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["optimizer"]["split_evaluations"] = split_evaluations

    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4,
        "types": [ConstraintType.LE],
    }
    enopt_config["realization_filters"] = [
        {
            "method": "sort-objective",
            "options": {
                "sort": [0],
                "first": 3,
                "last": 4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(
            enopt_config,
            evaluator(functions),
        )
        .add_observer(
            EventType.FINISHED_EVALUATION,
            partial(_track_results, result_list=result_list),
        )
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            assert result.evaluations.perturbed_constraints is not None
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_sort_filter_on_constraints(
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["optimizer"]["split_evaluations"] = split_evaluations
    enopt_config["nonlinear_constraints"] = {
        "names": ["c1"],
        "rhs_values": 0.4,
        "types": [ConstraintType.LE],
    }
    enopt_config["realization_filters"] = [
        {
            "method": "sort-constraint",
            "options": {
                "sort": "c1",
                "first": 3,
                "last": 4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(
            enopt_config,
            evaluator(functions),
        )
        .add_observer(
            EventType.FINISHED_EVALUATION,
            partial(_track_results, result_list=result_list),
        )
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            assert result.evaluations.perturbed_constraints is not None
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_sort_filter_mixed(  # noqa: C901
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["objective_functions"]["weights"] = [0.75, 0.25, 0.75, 0.25]
    enopt_config["objective_functions"]["names"] = ["f1", "f2", "f3", "f4"]
    enopt_config["optimizer"]["split_evaluations"] = split_evaluations

    objective_values: List[NDArray[np.float64]] = []

    def _add_objective(event: Event) -> None:
        if event.results:
            for item in event.results:
                if isinstance(item, FunctionResults):
                    assert item.functions is not None
                    objective_values.append(item.functions.weighted_objective)
        _track_results(event, result_list=result_list)

    # Apply the filtering to all objectives, giving the expected result.
    enopt_config["realization_filters"] = [
        {
            "method": "sort-objective",
            "options": {
                "sort": ["f1"],
                "first": 3,
                "last": 4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0, 0, 0]

    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(enopt_config, evaluator(functions))
        .add_observer(EventType.FINISHED_EVALUATION, _add_objective)
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )

    # Apply filtering only to the first two, giving a wrong result.
    enopt_config["realization_filters"] = [
        {
            "method": "sort-objective",
            "options": {
                "sort": [0],
                "first": 3,
                "last": 4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0, -1, -1]

    result_list = []
    results = (
        OptimizationPlanRunner(enopt_config, evaluator(functions))
        .add_observer(EventType.FINISHED_EVALUATION, _add_objective)
        .run()
        .results
    )
    assert results is not None
    assert not np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, :, :2] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, :, :2] != 0.0
                )
            assert np.all(
                result.evaluations.perturbed_objectives[filtered, :, 2:] != 0.0
            )

    # The first objective values should differ.
    assert objective_values[0] != objective_values[1]


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_cvar_filter_on_objectives(
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["optimizer"]["split_evaluations"] = split_evaluations

    variables = (
        OptimizationPlanRunner(enopt_config, evaluator(functions)).run().variables
    )
    assert variables is not None
    assert not np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

    enopt_config["realization_filters"] = [
        {
            "method": "cvar-objective",
            "options": {
                "sort": [0],
                "percentile": 0.4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0]
    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(
            enopt_config,
            evaluator(functions),
        )
        .add_observer(
            EventType.FINISHED_EVALUATION,
            partial(_track_results, result_list=result_list),
        )
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_cvar_filter_on_objectives_with_constraints(
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["optimizer"]["split_evaluations"] = split_evaluations

    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4,
        "types": [ConstraintType.LE],
    }
    enopt_config["realization_filters"] = [
        {
            "method": "cvar-objective",
            "options": {
                "sort": [0],
                "percentile": 0.4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(
            enopt_config,
            evaluator(functions),
        )
        .add_observer(
            EventType.FINISHED_EVALUATION,
            partial(_track_results, result_list=result_list),
        )
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            assert result.evaluations.perturbed_constraints is not None
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_cvar_filter_on_constraints(
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["optimizer"]["split_evaluations"] = split_evaluations

    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4,
        "types": [ConstraintType.LE],
    }
    enopt_config["realization_filters"] = [
        {
            "method": "cvar-constraint",
            "options": {
                "sort": 0,
                "percentile": 0.4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(
            enopt_config,
            evaluator(functions),
        )
        .add_observer(
            EventType.FINISHED_EVALUATION,
            partial(_track_results, result_list=result_list),
        )
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            assert result.evaluations.perturbed_constraints is not None
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )
                assert np.all(
                    result.evaluations.perturbed_constraints[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("split_evaluations", [True, False])
def test_cvar_filter_mixed(  # noqa: C901
    enopt_config: Any,
    evaluator: Any,
    split_evaluations: bool,
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["optimizer"]["split_evaluations"] = split_evaluations
    enopt_config["objective_functions"]["weights"] = [0.75, 0.25, 0.75, 0.25]
    enopt_config["objective_functions"]["names"] = ["f1", "f2", "f3", "f4"]

    objective_values: List[NDArray[np.float64]] = []

    def _add_objective(event: Event) -> None:
        if event.results is not None:
            for item in event.results:
                if isinstance(item, FunctionResults):
                    assert item.functions is not None
                    objective_values.append(item.functions.weighted_objective)
        _track_results(event, result_list=result_list)

    # Apply the filtering to all objectives, giving the expected result.
    enopt_config["realization_filters"] = [
        {
            "method": "cvar-objective",
            "options": {
                "sort": ["f1"],
                "percentile": 0.4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0, 0, 0]

    result_list: List[Results] = []
    results = (
        OptimizationPlanRunner(enopt_config, evaluator(functions))
        .add_observer(EventType.FINISHED_EVALUATION, _add_objective)
        .run()
        .results
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )

    # Apply filtering only to the first two, giving a wrong result.
    enopt_config["realization_filters"] = [
        {
            "method": "cvar-objective",
            "options": {
                "sort": [0],
                "percentile": 0.4,
            },
        },
    ]
    enopt_config["objective_functions"]["realization_filters"] = [0, 0, -1, -1]

    result_list = []
    results = (
        OptimizationPlanRunner(enopt_config, evaluator(functions))
        .add_observer(EventType.FINISHED_EVALUATION, _add_objective)
        .run()
        .results
    )
    assert results is not None
    assert not np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    for result in result_list:
        assert result is not None
        if isinstance(result, FunctionResults):
            assert result.realizations is not None
            assert result.realizations.objective_weights is not None
            filtered = result.realizations.objective_weights[0, :] == 0.0
        if isinstance(result, GradientResults):
            if split_evaluations:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, :, :2] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, :, :2] != 0.0
                )
            assert np.all(
                result.evaluations.perturbed_objectives[filtered, :, 2:] != 0.0
            )

    # The first objective values should differ.
    assert objective_values[0] != objective_values[1]
