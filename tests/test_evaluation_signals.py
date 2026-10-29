"""Tests for which results an evaluation returns and reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt._core import EnsembleEvaluator
from ropt.context import EnOptContext
from ropt.results import FunctionResults, GradientResults

if TYPE_CHECKING:
    from ropt.results import Results

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


class _Signals:
    def __init__(self) -> None:
        self.starts = 0
        self.reported: list[tuple[Results, ...]] = []

    def start(self) -> None:
        self.starts += 1

    def finish(self, results: tuple[Results, ...]) -> None:
        self.reported.append(results)


def _make(config: dict[str, Any], evaluator: Any) -> tuple[EnsembleEvaluator, _Signals]:
    signals = _Signals()
    ensemble = EnsembleEvaluator(
        EnOptContext.model_validate(config),
        evaluator(),
        signal_start=signals.start,
        signal_finished=signals.finish,
    )
    return ensemble, signals


def test_gradient_only_evaluation_returns_and_reports_the_function_result(
    config: Any, evaluator: Any
) -> None:
    ensemble, signals = _make(config, evaluator)
    results = ensemble.calculate(
        _INITIAL, compute_functions=False, compute_gradients=True
    )
    assert [type(item) for item in results] == [FunctionResults, GradientResults]
    assert signals.starts == 1
    assert signals.reported == [results]


def test_cached_function_results_are_returned_without_being_reported(
    config: Any, evaluator: Any
) -> None:
    ensemble, signals = _make(config, evaluator)
    first = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=False
    )
    second = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=False
    )
    assert second[0] is first[0]
    assert signals.starts == 1
    assert signals.reported == [first]


def test_close_function_point_is_evaluated_instead_of_returned_from_cache(
    config: Any, evaluator: Any
) -> None:
    ensemble, signals = _make(config, evaluator)
    first = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=False
    )
    close_point = _INITIAL.copy()
    close_point[0] = 5e-16
    second = ensemble.calculate(
        close_point, compute_functions=True, compute_gradients=False
    )
    repeated = ensemble.calculate(
        close_point.copy(), compute_functions=True, compute_gradients=False
    )

    assert second[0] is not first[0]
    assert isinstance(second[0], FunctionResults)
    np.testing.assert_array_equal(second[0].scaled.variables, close_point)
    assert repeated[0] is second[0]
    assert signals.starts == 2
    assert signals.reported == [first, second]


def test_gradient_evaluation_with_a_cached_function_reports_only_the_gradient(
    config: Any, evaluator: Any
) -> None:
    ensemble, signals = _make(config, evaluator)
    functions = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=False
    )
    gradients = ensemble.calculate(
        _INITIAL, compute_functions=False, compute_gradients=True
    )
    assert [type(item) for item in gradients] == [GradientResults]
    assert signals.starts == 2
    assert signals.reported == [functions, gradients]


def test_function_and_gradient_evaluation_reports_both_in_one_batch(
    config: Any, evaluator: Any
) -> None:
    ensemble, signals = _make(config, evaluator)
    results = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=True
    )
    assert [type(item) for item in results] == [FunctionResults, GradientResults]
    assert signals.starts == 1
    assert signals.reported == [results]


def test_function_and_gradient_evaluation_caches_the_function_result(
    config: Any, evaluator: Any
) -> None:
    ensemble, signals = _make(config, evaluator)
    results = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=True
    )
    again = ensemble.calculate(
        _INITIAL, compute_functions=True, compute_gradients=False
    )
    assert again[0] is results[0]
    assert signals.starts == 1
