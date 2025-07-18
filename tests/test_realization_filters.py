from functools import partial
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.plan import BasicOptimizer
from ropt.plugins.realization_filter.default import (
    _get_cvar_weights_from_percentile,
    _sort_and_select,
)
from ropt.results import FunctionResults, GradientResults, Results

initial_values = 3 * [0]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 10,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "number_of_perturbations": 5,
            "evaluation_policy": "speculative",
        },
        "realizations": {
            "weights": 5 * [1.0],
        },
        "variables": {
            "variable_count": 3,
            "perturbation_magnitudes": 0.01,
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
    failed: list[bool], first: int, last: int, expected: list[float]
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
    failed: list[bool], percentile: float, weights: list[float]
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
    realization: int,
    target: NDArray[np.float64],
) -> float:
    diff: NDArray[np.float64] = variables - target
    # Make sure that some realizations yield a different result:
    if realization % 2 == 0:
        diff += 1.0
    result = np.sum(diff**2)
    # Make sure the other realizations have a worse result:
    if realization % 2 == 1:
        result += 10.0
    return float(result)


def _constraint_function(variables: NDArray[np.float64], realization: int) -> float:
    # Track how often this function is called.
    result = variables[0] + variables[2]
    # Break some realizations, same as in the distance function:
    if realization % 2 == 0:
        result = variables[0] + variables[2] - 10.0
    return float(result)


def _track_results(results: tuple[Results, ...], result_list: list[Results]) -> None:
    result_list.extend(results)


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_sort_filter_on_objectives(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy

    variables = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).variables
    )
    assert variables is not None
    assert not np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

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
    enopt_config["objectives"]["realization_filters"] = [0, 0]

    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(partial(_track_results, result_list=result_list))
        .run(initial_values)
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
            if evaluation_policy == "separate":
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_sort_filter_on_objectives_with_constraints(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
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
    enopt_config["objectives"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(partial(_track_results, result_list=result_list))
        .run(initial_values)
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
            if evaluation_policy == "separate":
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


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_sort_filter_on_constraints(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    enopt_config["realization_filters"] = [
        {
            "method": "sort-constraint",
            "options": {
                "sort": 0,
                "first": 3,
                "last": 4,
            },
        },
    ]
    enopt_config["objectives"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(
            partial(_track_results, result_list=result_list),
        )
        .run(initial_values)
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
            if evaluation_policy == "separate":
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


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_sort_filter_mixed(  # noqa: C901
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["objectives"]["weights"] = [0.75, 0.25, 0.75, 0.25]
    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy

    objective_values: list[NDArray[np.float64]] = []

    def _add_objective(results: tuple[Results, ...]) -> None:
        if results:
            for item in results:
                if isinstance(item, FunctionResults):
                    assert item.functions is not None
                    objective_values.append(item.functions.weighted_objective)
        _track_results(results, result_list=result_list)

    # Apply the filtering to all objectives, giving the expected result.
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
    enopt_config["objectives"]["realization_filters"] = [0, 0, 0, 0]

    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(_add_objective)
        .run(initial_values)
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
            if evaluation_policy == "separate":
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
    enopt_config["objectives"]["realization_filters"] = [0, 0, -1, -1]

    result_list = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(_add_objective)
        .run(initial_values)
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

    # The first objective values should differ.
    assert objective_values[0] != objective_values[1]


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_cvar_filter_on_objectives(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy

    variables = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).variables
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
    enopt_config["objectives"]["realization_filters"] = [0, 0]
    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(
            partial(_track_results, result_list=result_list),
        )
        .run(initial_values)
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
            if evaluation_policy == "separate":
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] == 0.0
                )
            else:
                assert np.all(
                    result.evaluations.perturbed_objectives[filtered, ...] != 0.0
                )


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_cvar_filter_on_objectives_with_constraints(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
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
    enopt_config["objectives"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(
            partial(_track_results, result_list=result_list),
        )
        .run(initial_values)
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
            if evaluation_policy == "separate":
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


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_cvar_filter_on_constraints(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_constraint_function),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy

    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
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
    enopt_config["objectives"]["realization_filters"] = [0, 0]
    enopt_config["nonlinear_constraints"]["realization_filters"] = [0]
    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(
            partial(_track_results, result_list=result_list),
        )
        .run(initial_values)
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
            if evaluation_policy == "separate":
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


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_cvar_filter_mixed(
    enopt_config: Any,
    evaluator: Any,
    evaluation_policy: Literal["speculative", "separate", "auto"],
) -> None:
    functions = [
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
        partial(_objective_function, target=np.array([0.5, 0.5, 0.5])),
        partial(_objective_function, target=np.array([-1.5, -1.5, 0.5])),
    ]

    enopt_config["gradient"]["evaluation_policy"] = evaluation_policy
    enopt_config["objectives"]["weights"] = [0.75, 0.25, 0.75, 0.25]

    objective_values: list[NDArray[np.float64]] = []

    def _add_objective(results: tuple[Results, ...]) -> None:
        for item in results:
            if isinstance(item, FunctionResults):
                assert item.functions is not None
                objective_values.append(item.functions.weighted_objective)
        _track_results(results, result_list=result_list)

    # Apply the filtering to all objectives, giving the expected result.
    enopt_config["realization_filters"] = [
        {
            "method": "cvar-objective",
            "options": {
                "sort": [0],
                "percentile": 0.4,
            },
        },
    ]
    enopt_config["objectives"]["realization_filters"] = [0, 0, 0, 0]

    result_list: list[Results] = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(_add_objective)
        .run(initial_values)
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
            if evaluation_policy == "separate":
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
    enopt_config["objectives"]["realization_filters"] = [0, 0, -1, -1]

    result_list = []
    results = (
        BasicOptimizer(enopt_config, evaluator(functions))
        .set_results_callback(_add_objective)
        .run(initial_values)
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

    # The first objective values should differ.
    assert objective_values[0] != objective_values[1]
