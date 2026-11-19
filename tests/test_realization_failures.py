"""Tests for detecting failed realizations and reporting failed evaluations."""

from typing import Any, Literal, override

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt import optimize
from ropt._core import EnsembleEvaluator
from ropt._core._evaluator import (
    _get_failed_function_realizations,
    _get_failed_gradient_realizations,
)
from ropt._evaluation import EvaluationFunctionContext
from ropt._handlers import EventHandler, ResultsHandler
from ropt._running._run_evaluator import run_evaluation
from ropt.config import RealizationFilterConfig
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import TooFewRealizations
from ropt.realization_filter import RealizationFilter
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
    run_evaluator = evaluator([_square_failing_below_zero])
    handlers: list[EventHandler] = []
    handlers.append(handler)
    run_evaluation(
        evaluator=run_evaluator,
        handlers=handlers,
        context=context,
        variables=[[-1.0], [1.0], [2.0], [-1.0]],
    )
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
    _, gradient = EnsembleEvaluator(context, failing).calculate(
        np.zeros(2), compute_functions=True, compute_gradients=True
    )
    assert isinstance(gradient, GradientResults)
    assert gradient.target_gradient is not None
    assert gradient.gradients is not None
    assert gradient.gradients.constraints is not None
    assert np.all(np.isnan(gradient.target_gradient))
    assert np.all(np.isnan(gradient.gradients.constraints))


def _estimate(item: FunctionResults | GradientResults) -> object:
    return item.functions if isinstance(item, FunctionResults) else item.gradients


@pytest.mark.parametrize(
    ("evaluation_policy", "failing_rows", "expected"),
    [
        pytest.param(
            "speculative",
            "all",
            [("FunctionResults", True), ("GradientResults", True)],
            id="combined-batch-failing-in-the-function",
        ),
        pytest.param(
            "speculative",
            "perturbed",
            [("FunctionResults", False), ("GradientResults", True)],
            id="combined-batch-failing-in-the-gradient",
        ),
        pytest.param(
            "auto",
            "perturbed",
            [("FunctionResults", False), ("GradientResults", True)],
            id="gradient-batch",
        ),
    ],
)
def test_batch_with_a_failed_estimate_is_reported(
    evaluation_policy: str,
    failing_rows: Literal["all", "perturbed"],
    expected: list[tuple[str, bool]],
) -> None:
    # With one of the two realizations failing, the stddev estimator has too
    # few realizations to produce a value, which stops the run.
    def _function(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        failing = context.realization == 1 and (
            failing_rows == "all" or context.perturbation >= 0
        )
        return np.nan if failing else float(np.sum(variables**2)) + context.realization

    reported: list[FunctionResults | GradientResults] = []
    result = optimize(
        {
            "variables": {"variable_count": 2, "perturbation_magnitudes": 0.01},
            "objectives": {"weights": [1.0], "function_estimators": [0]},
            "function_estimators": [{"method": "stddev"}],
            "realizations": {"weights": [1.0, 1.0], "realization_min_success": 1},
            "gradient": {
                "number_of_perturbations": 2,
                "evaluation_policy": evaluation_policy,
            },
        },
        np.ones(2),
        _function,
        result_callback=reported.append,
        report_gradients=True,
    )
    assert result.exit_code == ExitCode.TOO_FEW_REALIZATIONS
    assert [
        (type(item).__name__, _estimate(item) is None) for item in reported
    ] == expected


class _FailingFilter(RealizationFilter):
    def __init__(self, _: RealizationFilterConfig) -> None:
        pass

    @override
    def get_realization_weights(
        self,
        objectives: NDArray[np.float64],
        _: NDArray[np.float64] | None,
        *,
        objective_scales: NDArray[np.float64],
        maximize: NDArray[np.bool_],
        objective_weights: NDArray[np.float64],
        constraint_lower_bounds: NDArray[np.float64] | None,
        constraint_upper_bounds: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        raise TooFewRealizations


def test_gradient_is_not_estimated_when_the_filter_fails(evaluator: Any) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 2},
            "objectives": {"weights": [1.0], "realization_filters": [0]},
            "realization_filters": [
                _FailingFilter(RealizationFilterConfig(method="failing"))
            ],
            "realizations": {"weights": [1.0, 1.0]},
        }
    )
    squares = evaluator([lambda variables, _: float(np.sum(variables**2))])
    function, gradient = EnsembleEvaluator(context, squares).calculate(
        np.zeros(2), compute_functions=True, compute_gradients=True
    )
    assert isinstance(function, FunctionResults)
    assert isinstance(gradient, GradientResults)
    assert function.functions is None
    assert gradient.gradients is None
