"""Tests for realization-failure detection in `ropt.core._evaluator`."""

from typing import Any

import numpy as np
import pytest

from ropt.core._evaluator import (
    _get_failed_function_realizations,
    _get_failed_gradient_realizations,
)
from ropt.results import (
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Realizations,
)


def test_failed_function_realization_nan_objective() -> None:
    failed_realizations = _get_failed_function_realizations(np.array([[np.nan, 1.0]]))
    assert np.all(failed_realizations == [True])


@pytest.mark.parametrize(
    ("objectives", "perturbed_objectives", "perturbation_min_success", "expected"),
    [
        pytest.param(
            np.array([[1.0, 1.0]]),
            np.array([[5 * [np.nan, 1.0]]]),
            1,
            [True],
            id="no_perturbation_success",
        ),
        pytest.param(
            np.array([[1.0, 1.0]]),
            np.array(
                [
                    [
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [np.nan, 1.0],
                        [np.nan, 1.0],
                        [np.nan, np.nan],
                    ]
                ]
            ),
            4,
            [True],
        ),
        pytest.param(
            np.array([[1.0, 1.0]]),
            np.array([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [np.nan, 1.0]]]),
            4,
            [False],
        ),
        pytest.param(
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[[1.0, 1.0]], [[np.nan, 1.0]]]),
            1,
            [False, True],
        ),
        pytest.param(
            np.array([[1.0, 1.0], [np.nan, 1.0]]),
            np.array([[[1.0], [1.0]], [[1.0], [1.0]]]),
            1,
            [False, True],
        ),
        pytest.param(
            np.array([[1.0, 1.0], [np.nan, np.nan]]),
            np.array([[[1.0], [1.0]], [[1.0], [1.0]]]),
            1,
            [False, True],
        ),
        pytest.param(
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[[1.0, 1.0], [1.0, 1.0]], [[np.nan, np.nan], [1.0, 1.0]]]),
            1,
            [False, False],
        ),
    ],
)
def test_failed_gradient_realizations(
    objectives: Any,
    perturbed_objectives: Any,
    perturbation_min_success: int,
    expected: list[bool],
) -> None:
    failed_realizations = _get_failed_gradient_realizations(
        objectives, perturbed_objectives, perturbation_min_success
    )
    assert np.all(failed_realizations == expected)


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {"variable_count": 2},
        "objectives": {"weights": [1.0]},
        "realizations": {
            "weights": [1.0, 1.0, 1.0],
            "realization_min_success": 2,
        },
        "gradient": {
            "number_of_perturbations": 4,
            "perturbation_min_success": 3,
        },
    }


def _make_function_results(*, failed: bool) -> FunctionResults:
    evaluations = FunctionEvaluations.create(
        variables=np.array([0.0, 0.0]),
        objectives=(
            np.array([[np.nan], [np.nan], [np.nan]], dtype=np.float64)
            if failed
            else np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
        ),
    )
    return FunctionResults(
        batch_id=0,
        metadata={},
        names={},
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(3, dtype=np.bool_),
        ),
        functions=(
            None
            if failed
            else Functions.create(
                target_objective=np.array(1.0),
                objectives=np.array([1.0]),
            )
        ),
    )


def _make_gradient_results(
    *, failed: bool, perturbation_failures: bool
) -> GradientResults:
    if perturbation_failures:
        perturbed_objectives = np.full((3, 4, 1), np.nan, dtype=np.float64)
        perturbed_objectives[:, 0, 0] = 1.0
    else:
        perturbed_objectives = np.ones((3, 4, 1), dtype=np.float64)
    evaluations = GradientEvaluations(
        variables=np.array([0.0, 0.0]),
        perturbed_variables=np.zeros((3, 4, 2), dtype=np.float64),
        perturbed_objectives=perturbed_objectives,
        metadata={},
    )
    return GradientResults(
        batch_id=0,
        metadata={},
        names={},
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(3, dtype=np.bool_),
        ),
        gradients=None if failed else object(),  # type: ignore[arg-type]
    )
