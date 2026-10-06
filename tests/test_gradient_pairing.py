"""Tests for pairing a gradient with the function results it came from."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ropt import optimize
from ropt.components.event_handlers import ResultsHandler
from ropt.context import EnOptContext
from ropt.core import EnsembleEvaluator
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import (
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Realizations,
    ScaledFunctionResults,
    ScaledGradientResults,
)

_INITIAL = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": _INITIAL.size,
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {"weights": [1.0, 1.0]},
        "realizations": {"weights": [1.0, 1.0]},
        "gradient": {"number_of_perturbations": 3},
    }


def _ensemble(config: dict[str, Any], evaluator: Any) -> EnsembleEvaluator:
    return EnsembleEvaluator(EnOptContext.model_validate(config), evaluator().eval)


def test_gradient_of_a_combined_batch_keys_the_function_beside_it(
    config: Any, evaluator: Any
) -> None:
    ensemble = _ensemble(config, evaluator)
    function, gradient = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=True
    )
    assert isinstance(function, FunctionResults)
    assert isinstance(gradient, GradientResults)
    assert gradient.function_key == function.function_key
    assert gradient.function_key == (function.batch_id, 0)


def test_gradient_of_a_cached_function_keys_the_earlier_batch(
    config: Any, evaluator: Any
) -> None:
    ensemble = _ensemble(config, evaluator)
    functions = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=False
    )
    gradients = ensemble.calculate(
        _INITIAL, compute_functions=False, compute_gradients=True
    )
    function = functions[0]
    gradient = gradients[0]
    assert isinstance(function, FunctionResults)
    assert isinstance(gradient, GradientResults)
    assert gradient.function_key == function.function_key
    # The pairing is what the batch id cannot express here.
    assert gradient.batch_id != gradient.function_key[0]


def test_function_results_of_one_batch_have_distinct_keys(
    config: Any, evaluator: Any
) -> None:
    ensemble = _ensemble(config, evaluator)
    variables = np.repeat(_INITIAL[np.newaxis, :], 3, axis=0)
    results = ensemble.calculate(
        variables, compute_functions=True, compute_gradients=False
    )
    assert all(isinstance(item, FunctionResults) for item in results)
    functions = cast("tuple[FunctionResults, ...]", results)
    assert [item.function_id for item in functions] == [0, 1, 2]
    assert len({item.function_key for item in functions}) == 3


def _function_results(batch_id: int, objective: float) -> FunctionResults:
    return FunctionResults(
        batch_id=batch_id,
        function_id=0,
        metadata={},
        names={},
        variables=np.zeros(1),
        evaluations=FunctionEvaluations.create(objectives=np.array([[objective]])),
        realizations=Realizations(evaluated_realizations=np.ones(1, dtype=np.bool_)),
        functions=Functions(objectives=np.array([objective])),
        target_objective=np.array(objective),
        scaled=ScaledFunctionResults(variables=np.zeros(1)),
    )


def _gradient_results(batch_id: int, function_key: tuple[int, int]) -> GradientResults:
    return GradientResults(
        batch_id=batch_id,
        function_key=function_key,
        metadata={},
        names={},
        variables=np.zeros(1),
        perturbed_variables=np.zeros((1, 1, 1)),
        evaluations=GradientEvaluations.create(
            perturbed_objectives=np.zeros((1, 1, 1))
        ),
        realizations=Realizations(evaluated_realizations=np.ones(1, dtype=np.bool_)),
        gradients=None,
        target_gradient=None,
        scaled=ScaledGradientResults(
            variables=np.zeros(1), perturbed_variables=np.zeros((1, 1, 1))
        ),
    )


def _event(*results: Any) -> EnOptEvent:
    return EnOptEvent(
        event_type=EnOptEventType.FINISHED_EVALUATION,
        context=None,  # type: ignore[arg-type]
        results=results,
    )


def test_results_handler_keeps_a_gradient_arriving_in_a_later_event() -> None:
    handler = ResultsHandler()
    results = _function_results(batch_id=0, objective=2.0)
    gradient = _gradient_results(batch_id=1, function_key=results.function_key)
    handler.handle_event(_event(results))
    assert handler["gradient"] is None
    handler.handle_event(_event(gradient))
    assert handler["gradient"] is gradient


def test_results_handler_drops_the_gradient_when_the_best_changes() -> None:
    handler = ResultsHandler()
    first = _function_results(batch_id=0, objective=2.0)
    handler.handle_event(_event(first, _gradient_results(1, first.function_key)))
    assert handler["gradient"] is not None
    better = _function_results(batch_id=2, objective=1.0)
    handler.handle_event(_event(better))
    assert handler["results"] is better
    assert handler["gradient"] is None


def test_results_handler_ignores_a_gradient_of_another_result() -> None:
    handler = ResultsHandler()
    best = _function_results(batch_id=0, objective=1.0)
    handler.handle_event(_event(best))
    worse = _function_results(batch_id=2, objective=3.0)
    handler.handle_event(_event(worse, _gradient_results(3, worse.function_key)))
    assert handler["results"] is best
    assert handler["gradient"] is None


def test_results_handler_clearing_the_result_clears_the_gradient() -> None:
    handler = ResultsHandler()
    results = _function_results(batch_id=0, objective=2.0)
    handler.handle_event(_event(results, _gradient_results(1, results.function_key)))
    assert handler["gradient"] is not None
    handler["results"] = None
    handler.handle_event(_event(_function_results(batch_id=2, objective=5.0)))
    assert handler["gradient"] is None


def test_optimize_returns_the_gradient_at_the_best_result(
    config: Any, test_functions: Any
) -> None:
    config["optimizer"] = {"max_functions": 4}
    config["objectives"] = {"weights": [1.0]}
    result = optimize(config, _INITIAL, test_functions[0])
    assert result.results is not None
    assert result.gradient is not None
    assert result.gradient.function_key == result.results.function_key


def test_report_receives_gradients_only_when_asked(
    config: Any, test_functions: Any
) -> None:
    config["optimizer"] = {"max_functions": 3}
    config["objectives"] = {"weights": [1.0]}

    def _collect(reported: list[Any]) -> Any:
        def _report(item: Any) -> None:
            reported.append(type(item))

        return _report

    default: list[Any] = []
    optimize(config, _INITIAL, test_functions[0], report=_collect(default))
    assert set(default) == {FunctionResults}

    both: list[Any] = []
    optimize(
        config,
        _INITIAL,
        test_functions[0],
        report=_collect(both),
        report_gradients=True,
    )
    assert set(both) == {FunctionResults, GradientResults}
