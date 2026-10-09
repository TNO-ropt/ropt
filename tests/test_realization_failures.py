"""Tests for detecting failed realizations and reporting failed evaluations."""

from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.components.compute_steps import EvaluationStep
from ropt.components.evaluators import EvaluationFunctionContext
from ropt.components.event_handlers import ResultsHandler
from ropt.context import EnOptContext
from ropt.core import EnsembleEvaluator
from ropt.core._evaluator import (
    _get_failed_function_realizations,
    _get_failed_gradient_realizations,
)
from ropt.results import (
    ConstraintInfo,
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Realizations,
    ScaledFunctionResults,
    ScaledGradientResults,
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
        objectives=(
            np.array([[np.nan], [np.nan], [np.nan]], dtype=np.float64)
            if failed
            else np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
        ),
    )
    return FunctionResults(
        batch_id=0,
        function_id=0,
        metadata={},
        names={},
        variables=np.array([0.0, 0.0]),
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(3, dtype=np.bool_),
        ),
        functions=None if failed else Functions(objectives=np.array([1.0])),
        target_objective=None if failed else np.array(1.0),
        scaled=ScaledFunctionResults(
            variables=np.array([0.0, 0.0]),
            functions=None if failed else Functions(objectives=np.array([1.0])),
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
        perturbed_objectives=perturbed_objectives,
        metadata={},
    )
    return GradientResults(
        batch_id=0,
        source_key=(0, 0),
        metadata={},
        names={},
        variables=np.array([0.0, 0.0]),
        perturbed_variables=np.zeros((3, 4, 2), dtype=np.float64),
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(3, dtype=np.bool_),
        ),
        gradients=None if failed else object(),  # type: ignore[arg-type]
        target_gradient=None if failed else np.zeros(2),
        scaled=ScaledGradientResults(
            variables=np.array([0.0, 0.0]),
            perturbed_variables=np.zeros((3, 4, 2), dtype=np.float64),
            gradients=None if failed else object(),  # type: ignore[arg-type]
        ),
    )


def _square_failing_below_zero(
    variables: NDArray[np.float64], _: EvaluationFunctionContext
) -> float:
    return np.nan if variables[0] < 0.0 else float(variables[0] ** 2)


@pytest.mark.parametrize(("what", "target"), [("best", 1.0), ("last", 4.0)])
def test_results_handler_skips_an_evaluation_where_all_realizations_failed(
    evaluator: Any, what: Literal["best", "last"], target: float
) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 1},
            "realizations": {"realization_min_success": 0},
        }
    )
    handler = ResultsHandler(what=what)
    step = EvaluationStep(evaluator=evaluator([_square_failing_below_zero]))
    step.add_event_handler(handler)
    step.run(context=context, variables=[[-1.0], [1.0], [2.0], [-1.0]])
    assert handler.result is not None
    assert handler.result.target_objective == target


@pytest.mark.parametrize(
    ("lower", "upper", "violation"),
    [
        pytest.param(np.nan, np.nan, np.nan, id="nan-value"),
        pytest.param(np.nan, -np.inf, 0.0, id="infinite-value-at-infinite-bound"),
        pytest.param(-1.5, -3.5, 1.5, id="value-below-lower-bound"),
    ],
)
def test_constraint_info_violation_is_nan_only_for_a_nan_value(
    lower: float, upper: float, violation: float
) -> None:
    info = ConstraintInfo(
        nonlinear_lower=np.array([lower]), nonlinear_upper=np.array([upper])
    )
    assert info.nonlinear_violation is not None
    assert np.array_equal(info.nonlinear_violation, [violation], equal_nan=True)


@pytest.mark.parametrize("merge_realizations", [False, True])
def test_gradient_is_nan_when_every_realization_failed(
    evaluator: Any, *, merge_realizations: bool
) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 2},
            "realizations": {"weights": [1.0, 1.0], "realization_min_success": 0},
            "nonlinear_constraints": {"lower_bounds": [0.0], "upper_bounds": [1.0]},
            "gradient": {"merge_realizations": merge_realizations},
        }
    )
    failing = evaluator([lambda _0, _1: np.nan], [lambda _0, _1: np.nan])
    _, gradient = EnsembleEvaluator(context, failing.eval).calculate(
        np.zeros(2), compute_functions=True, compute_gradients=True
    )
    assert isinstance(gradient, GradientResults)
    assert gradient.target_gradient is not None
    assert gradient.gradients is not None
    assert gradient.gradients.constraints is not None
    assert np.all(np.isnan(gradient.target_gradient))
    assert np.all(np.isnan(gradient.gradients.constraints))
