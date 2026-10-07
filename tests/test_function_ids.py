"""Tests for the function evaluation index carried by an evaluation batch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.context import EnOptContext
from ropt.core import EnsembleEvaluator
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

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


class _Recorder:
    def __init__(self) -> None:
        self.function_ids: list[NDArray[np.intc]] = []

    def eval(
        self, variables: NDArray[np.float64], context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        self.function_ids.append(context.function_ids)
        return EvaluationBatchResult(objectives=np.ones((variables.shape[0], 2)))


def _ensemble(config: dict[str, Any], recorder: _Recorder) -> EnsembleEvaluator:
    return EnsembleEvaluator(EnOptContext.model_validate(config), recorder.eval)


def test_function_rows_are_numbered_by_variable_vector(config: Any) -> None:
    recorder = _Recorder()
    variables = np.repeat(_INITIAL[np.newaxis, :], 3, axis=0)
    _ensemble(config, recorder).calculate(
        variables, compute_functions=True, compute_gradients=False
    )
    assert np.array_equal(recorder.function_ids[0], [0, 0, 1, 1, 2, 2])


def test_gradient_rows_evaluate_no_function(config: Any) -> None:
    recorder = _Recorder()
    ensemble = _ensemble(config, recorder)
    ensemble.calculate(_INITIAL, compute_functions=True, compute_gradients=False)
    ensemble.calculate(_INITIAL, compute_functions=False, compute_gradients=True)
    assert np.array_equal(recorder.function_ids[1], [-1] * 6)


def test_combined_batch_numbers_only_its_function_rows(config: Any) -> None:
    recorder = _Recorder()
    _ensemble(config, recorder).calculate(
        _INITIAL, compute_functions=True, compute_gradients=True
    )
    assert np.array_equal(recorder.function_ids[0], [0, 0, *([-1] * 6)])
