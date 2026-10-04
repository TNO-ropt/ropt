"""Tests for which results an evaluation returns and reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.context import EnOptContext
from ropt.core import EnsembleEvaluator
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
        self.calls: list[tuple[Results, ...] | None] = []

    def __call__(self, results: tuple[Results, ...] | None = None) -> None:
        self.calls.append(results)

    @property
    def starts(self) -> int:
        return sum(item is None for item in self.calls)

    @property
    def reported(self) -> list[tuple[Results, ...]]:
        return [item for item in self.calls if item is not None]


def _make(config: dict[str, Any], evaluator: Any) -> tuple[EnsembleEvaluator, _Signals]:
    signals = _Signals()
    ensemble = EnsembleEvaluator(
        EnOptContext.model_validate(config),
        evaluator().eval,
        signal_evaluation=signals,
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
