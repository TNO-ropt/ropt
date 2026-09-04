"""Tests for the scaling and direction of objectives and constraints.

Objectives and constraints are divided by their scale before they reach the
optimizer and multiplied by it again before they are reported. Scales are
positive: a scale is a change of units only. Direction is separate, set per
objective by `maximize`, and applied to aggregated objectives alone.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.components.event_handlers import CallbackHandler
from ropt.context import EnOptContext
from ropt.core._auto_scale import set_auto_scales
from ropt.enums import EnOptEventType
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.events import EnOptEvent
from ropt.results import (
    ConstraintInfo,
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientResults,
    Gradients,
    Realizations,
    Results,
)
from ropt.simple import optimize


def _context(**fields: Any) -> EnOptContext:
    return EnOptContext.model_validate({"variables": {"variable_count": 2}} | fields)


def _function_results(
    objectives: NDArray[np.float64] | None = None,
    constraints: NDArray[np.float64] | None = None,
    functions: Functions | None = None,
    constraint_info: ConstraintInfo | None = None,
) -> FunctionResults:
    return FunctionResults(
        batch_id=0,
        metadata={},
        names={},
        evaluation_point=np.zeros(2),
        evaluations=FunctionEvaluations(
            variables=np.zeros(2),
            objectives=np.zeros((1, 2)) if objectives is None else objectives,
            constraints=constraints,
        ),
        realizations=Realizations(evaluated_realizations=np.array([True])),
        functions=functions,
        constraint_info=constraint_info,
    )


def test_objective_scales_are_undone_when_reporting() -> None:
    context = _context(objectives={"weights": [0.5, 0.5], "scales": [2.0, 4.0]})
    results = _function_results(objectives=np.array([[3.0, 5.0]]))
    transformed = results.transform_from_optimizer(context)
    assert np.allclose(transformed.evaluations.objectives, [[6.0, 20.0]])


def test_per_realization_objectives_are_never_flipped() -> None:
    context = _context(
        objectives={"weights": [0.5, 0.5], "scales": [2.0, 4.0], "maximize": True}
    )
    results = _function_results(objectives=np.array([[3.0, 5.0]]))
    transformed = results.transform_from_optimizer(context)
    # Direction applies to aggregates only, so these are scaled and nothing else.
    assert np.allclose(transformed.evaluations.objectives, [[6.0, 20.0]])


def test_constraint_scales_are_undone_when_reporting() -> None:
    context = _context(
        nonlinear_constraints={
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [1.0, 1.0],
            "scales": [2.0, 4.0],
        }
    )
    results = _function_results(constraints=np.array([[3.0, 5.0]]))
    transformed = results.transform_from_optimizer(context)
    assert transformed.evaluations.constraints is not None
    assert np.allclose(transformed.evaluations.constraints, [[6.0, 20.0]])


def test_constraint_bounds_keep_their_order_when_scaled() -> None:
    context = _context(
        nonlinear_constraints={
            "lower_bounds": [1.0, 1.0],
            "upper_bounds": [4.0, 4.0],
            "scales": [2.0, 4.0],
        }
    )
    bounds = context.get_nonlinear_constraint_bounds()
    assert bounds is not None
    lower, upper = bounds
    assert np.allclose(lower, [0.5, 0.25])
    assert np.allclose(upper, [2.0, 1.0])
    assert np.all(lower <= upper)


def test_constraint_diffs_are_scaled_back() -> None:
    context = _context(
        nonlinear_constraints={
            "lower_bounds": [1.0, 1.0],
            "upper_bounds": [4.0, 4.0],
            "scales": [2.0, 4.0],
        }
    )
    results = _function_results(
        constraint_info=ConstraintInfo(
            nonlinear_lower=np.array([0.25, 0.5]),
            nonlinear_upper=np.array([-0.25, -0.5]),
        )
    )
    transformed = results.transform_from_optimizer(context)
    assert transformed.constraint_info is not None
    assert transformed.constraint_info.nonlinear_lower is not None
    assert transformed.constraint_info.nonlinear_upper is not None
    assert np.allclose(transformed.constraint_info.nonlinear_lower, [0.5, 2.0])
    assert np.allclose(transformed.constraint_info.nonlinear_upper, [-0.5, -2.0])


def test_the_direction_of_an_aggregate_is_undone_when_reporting() -> None:
    context = _context(
        objectives={
            "weights": [0.5, 0.5],
            "scales": [2.0, 2.0],
            "maximize": [False, True],
        }
    )
    results = _function_results(
        functions=Functions(
            target_objective=np.array(1.0),
            objectives=np.array([3.0, -5.0]),
        )
    )
    transformed = results.transform_from_optimizer(context)
    assert transformed.functions is not None
    # The maximized objective was negated for the optimizer; reporting it
    # undoes that, so it comes back positive.
    assert np.allclose(transformed.functions.objectives, [6.0, 10.0])


def test_the_target_objective_stays_in_the_optimizer_domain() -> None:
    context = _context(
        objectives={"weights": [0.5, 0.5], "scales": [2.0, 2.0], "maximize": True}
    )
    results = _function_results(
        functions=Functions(
            target_objective=np.array(7.0),
            objectives=np.array([3.0, 5.0]),
        )
    )
    transformed = results.transform_from_optimizer(context)
    assert transformed.functions is not None
    # It mixes objectives of different scales and directions, so there is no
    # single factor to undo: it is always a value to minimize.
    assert np.allclose(transformed.functions.target_objective, 7.0)


def test_gradients_are_scaled_as_differences() -> None:
    context = _context(objectives={"weights": [0.5, 0.5], "scales": [2.0, 4.0]})
    gradients = Gradients(
        target_objective=np.zeros(2),
        objectives=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    transformed = gradients._transform_from_optimizer(context)  # ruff: ignore[private-member-access]
    assert transformed is not None
    # Every column of a row is scaled by the scale of that objective.
    assert np.allclose(transformed.objectives, [[2.0, 4.0], [12.0, 16.0]])


def test_gradient_directions_are_undone_when_reporting() -> None:
    context = _context(
        objectives={
            "weights": [0.5, 0.5],
            "scales": [2.0, 4.0],
            "maximize": [False, True],
        }
    )
    gradients = Gradients(
        target_objective=np.zeros(2),
        objectives=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    transformed = gradients._transform_from_optimizer(context)  # ruff: ignore[private-member-access]
    assert transformed is not None
    assert np.allclose(transformed.objectives, [[2.0, 4.0], [-12.0, -16.0]])


def test_scales_default_to_one() -> None:
    context = _context(objectives={"weights": [0.5, 0.5]})
    assert np.allclose(context.get_objective_scales(), 1.0)
    assert context.get_constraint_scales() is None


# Auto-scaling estimates a factor from the first batch of evaluations. Two
# realizations with weights 1 and 3 are used throughout, so that a plain average
# and a weighted one give different answers.


def _auto_scale_context(**fields: Any) -> EnOptContext:
    return EnOptContext.model_validate(
        {
            "variables": {"variable_count": 2},
            "realizations": {"weights": [1.0, 3.0]},
        }
        | fields
    )


def _estimate(
    context: EnOptContext,
    objectives: NDArray[np.float64],
    constraints: NDArray[np.float64] | None = None,
    perturbations: NDArray[np.intc] | None = None,
    active: NDArray[np.bool_] | None = None,
) -> None:
    rows = objectives.shape[0]
    set_auto_scales(
        context,
        EvaluationBatchContext(
            context=context,
            active=np.ones(rows, dtype=np.bool_) if active is None else active,
            realizations=np.arange(rows, dtype=np.intc) % 2,
            perturbations=perturbations,
        ),
        EvaluationBatchResult(objectives=objectives, constraints=constraints),
    )


def test_auto_scale_uses_a_single_factor_for_all_objectives() -> None:
    context = _auto_scale_context(
        objectives={"weights": [0.25, 0.75], "auto_scale": True}
    )
    _estimate(context, np.array([[2.0, 6.0], [2.0, 6.0]]))
    # A single factor, the weighted total: 0.25 * 2 + 0.75 * 6.
    assert np.allclose(context.get_objective_scales(), 5.0)


def test_auto_scale_preserves_the_relative_size_of_the_objectives() -> None:
    context = _auto_scale_context(
        objectives={"weights": [0.5, 0.5], "auto_scale": True}
    )
    _estimate(context, np.array([[2.0, 6.0], [2.0, 6.0]]))
    scaled = np.array([2.0, 6.0]) / context.get_objective_scales()
    assert np.allclose(scaled[1] / scaled[0], 3.0)


def test_auto_scale_uses_a_factor_per_constraint() -> None:
    context = _auto_scale_context(
        objectives={"weights": [0.5, 0.5], "auto_scale": True},
        nonlinear_constraints={
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [1.0, 1.0],
            "auto_scale": True,
        },
    )
    _estimate(
        context,
        np.array([[2.0, 6.0], [2.0, 6.0]]),
        constraints=np.array([[5.0, -10.0], [5.0, -10.0]]),
    )
    # Each constraint gets its own factor, and the sign is dropped.
    constraint_scales = context.get_constraint_scales()
    assert constraint_scales is not None
    assert np.allclose(constraint_scales, [5.0, 10.0])


def test_auto_scale_skips_the_constraints_that_are_not_flagged() -> None:
    context = _auto_scale_context(
        objectives={"weights": [0.5, 0.5]},
        nonlinear_constraints={
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [1.0, 1.0],
            "scales": [2.0, 3.0],
            "auto_scale": [True, False],
        },
    )
    # The estimate of the second constraint vanishes, but it is not inspected.
    _estimate(
        context,
        np.array([[2.0, 6.0], [2.0, 6.0]]),
        constraints=np.array([[5.0, 0.0], [5.0, 0.0]]),
    )
    constraint_scales = context.get_constraint_scales()
    assert constraint_scales is not None
    assert np.allclose(constraint_scales, [10.0, 3.0])


def test_auto_scale_weighs_the_realizations() -> None:
    context = _auto_scale_context(
        objectives={"weights": [1.0], "auto_scale": True},
    )
    # Realization 0 has weight 1, realization 1 has weight 3.
    _estimate(context, np.array([[4.0], [8.0]]))
    assert np.allclose(context.get_objective_scales(), (4.0 + 3 * 8.0) / 4)


def test_auto_scale_ignores_perturbed_rows() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    _estimate(
        context,
        np.array([[4.0], [8.0], [1000.0], [1000.0]]),
        perturbations=np.array([-1, -1, 0, 0], dtype=np.intc),
    )
    assert np.allclose(context.get_objective_scales(), (4.0 + 3 * 8.0) / 4)


def test_auto_scale_falls_back_to_perturbed_rows_when_there_are_no_others() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    _estimate(
        context,
        np.array([[4.0], [8.0]]),
        perturbations=np.array([0, 0], dtype=np.intc),
    )
    assert np.allclose(context.get_objective_scales(), (4.0 + 3 * 8.0) / 4)


def test_auto_scale_ignores_rows_that_were_not_evaluated() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    # An inactive row holds a zero that was never computed.
    _estimate(
        context,
        np.array([[4.0], [8.0], [0.0], [0.0]]),
        active=np.array([True, True, False, False]),
    )
    assert np.allclose(context.get_objective_scales(), (4.0 + 3 * 8.0) / 4)


def test_auto_scale_ignores_failed_realizations() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    _estimate(context, np.array([[4.0], [8.0], [np.nan], [np.nan]]))
    assert np.allclose(context.get_objective_scales(), (4.0 + 3 * 8.0) / 4)


def test_auto_scale_composes_with_a_configured_scale() -> None:
    context = _auto_scale_context(
        objectives={"weights": [1.0], "scales": [3.0], "auto_scale": True}
    )
    _estimate(context, np.array([[4.0], [8.0]]))
    # The estimate multiplies the configured scale rather than replacing it.
    assert np.allclose(context.get_objective_scales(), 3.0 * (4.0 + 3 * 8.0) / 4)


def test_auto_scale_leaves_the_scales_alone_until_it_runs() -> None:
    context = _auto_scale_context(
        objectives={"weights": [1.0], "scales": [2.0], "auto_scale": True}
    )
    assert np.allclose(context.get_objective_scales(), 2.0)


def test_auto_scale_stays_positive() -> None:
    context = _auto_scale_context(
        objectives={"weights": [1.0], "scales": [3.0], "auto_scale": True},
    )
    # The objectives are negative, but a scale is a change of units.
    _estimate(context, np.array([[-4.0], [-8.0]]))
    assert np.all(context.get_objective_scales() > 0.0)


def test_auto_scale_uses_the_first_batch_only() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    _estimate(context, np.array([[4.0], [8.0]]))
    first = context.get_objective_scales()
    _estimate(context, np.array([[400.0], [800.0]]))
    assert np.allclose(context.get_objective_scales(), first)


def test_auto_scale_updates_the_constraint_bounds() -> None:
    context = _auto_scale_context(
        objectives={"weights": [1.0]},
        nonlinear_constraints={
            "lower_bounds": [0.0],
            "upper_bounds": [10.0],
            "auto_scale": True,
        },
    )
    bounds = context.get_nonlinear_constraint_bounds()
    assert bounds is not None
    assert np.allclose(bounds[1], 10.0)
    _estimate(context, np.array([[1.0], [1.0]]), constraints=np.array([[5.0], [5.0]]))
    bounds = context.get_nonlinear_constraint_bounds()
    assert bounds is not None
    assert np.allclose(bounds[1], 2.0)


def test_auto_scale_rejects_a_vanishing_estimate() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    with pytest.raises(RuntimeError, match="failed to estimate a scale factor"):
        _estimate(context, np.zeros((2, 1)))


def test_auto_scale_rejects_a_batch_without_results() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    with pytest.raises(RuntimeError, match="no realizations to average"):
        _estimate(context, np.full((2, 1), np.nan))


def test_auto_scale_is_skipped_when_it_is_not_configured() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "scales": [2.0]})
    _estimate(context, np.array([[4.0], [8.0]]))
    assert np.allclose(context.get_objective_scales(), 2.0)


def test_the_estimated_scales_cannot_be_set_twice() -> None:
    context = _auto_scale_context(objectives={"weights": [1.0], "auto_scale": True})
    context._set_auto_scales(np.array(2.0), None)  # ruff: ignore[private-member-access]
    with pytest.raises(RuntimeError, match="already been set"):
        context._set_auto_scales(np.array(2.0), None)  # ruff: ignore[private-member-access]


# The invariant that separating scale from direction buys: in the user domain
# the reported aggregate equals the aggregate of the reported per-realization
# values. A negative scale used to break this for a spread, which came back
# positive while the values it summarized came back negative.


def _run(objectives: dict[str, Any], estimator: str) -> list[Results]:
    collected: list[Results] = []

    def collect(event: EnOptEvent) -> None:
        # Results reach a handler in the optimizer domain; reporting them is
        # what maps them back.
        collected.extend(
            item.transform_from_optimizer(event.context)
            for item in (event.results or ())
        )

    optimize(
        {
            "optimizer": {"max_functions": 1},
            "variables": {"variable_count": 2, "perturbation_magnitudes": 0.01},
            "realizations": {"weights": [1.0, 2.0, 3.0]},
            "objectives": objectives,
            "gradient": {"evaluation_policy": "speculative"},
            "function_estimators": {"0": {"method": estimator}},
        },
        np.array([0.5, 0.5]),
        _spread_evaluator,
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION}, callback=collect
            )
        ],
    )
    return collected


def _run_and_collect(
    objectives: dict[str, Any], estimator: str
) -> list[FunctionResults]:
    return [
        item
        for item in _run(objectives, estimator)
        if isinstance(item, FunctionResults)
    ]


def _first_gradient(objectives: dict[str, Any], estimator: str) -> Gradients:
    for item in _run(objectives, estimator):
        if isinstance(item, GradientResults) and item.gradients is not None:
            return item.gradients
    msg = "the run produced no gradients"
    raise AssertionError(msg)


def _spread_evaluator(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    # One objective, varying across realizations so that a standard deviation
    # is non-zero.
    return float(context.realization + 1) * (1.0 + float(variables.sum()))


@pytest.mark.parametrize("estimator", ["mean", "stddev"])
@pytest.mark.parametrize("maximize", [False, True])
def test_the_reported_aggregate_matches_the_reported_values(
    estimator: str, *, maximize: bool
) -> None:
    results = _run_and_collect(
        {"weights": [1.0], "scales": [4.0], "maximize": maximize}, estimator
    )
    assert results

    context = _context(
        realizations={"weights": [1.0, 2.0, 3.0]},
        objectives={"weights": [1.0]},
        function_estimators={"0": {"method": estimator}},
    )
    aggregate = context.function_estimators["0"]
    aggregate.init(context)

    for item in results:
        assert item.functions is not None
        values = item.evaluations.objectives[:, 0]
        assert np.allclose(
            item.functions.objectives[0],
            aggregate.calculate_function(values, context.realizations.weights),
        )


@pytest.mark.parametrize("estimator", ["mean", "stddev"])
def test_maximizing_negates_what_the_optimizer_minimizes(estimator: str) -> None:
    minimized = _run_and_collect({"weights": [1.0]}, estimator)
    maximized = _run_and_collect({"weights": [1.0], "maximize": True}, estimator)
    assert minimized
    assert maximized
    for low, high in zip(minimized, maximized, strict=True):
        assert low.functions is not None
        assert high.functions is not None
        # `target_objective` is reported in the optimizer domain, so the flip
        # shows there. A spread flips too, which sign-blind aggregation of
        # negated inputs would not have achieved.
        assert np.allclose(
            high.functions.target_objective, -low.functions.target_objective
        )
        assert low.functions.target_objective > 0.0


@pytest.mark.parametrize("estimator", ["mean", "stddev"])
def test_maximizing_negates_the_gradient_the_optimizer_follows(estimator: str) -> None:
    minimized = _first_gradient({"weights": [1.0]}, estimator)
    maximized = _first_gradient({"weights": [1.0], "maximize": True}, estimator)
    # `target_objective` is the gradient the optimizer descends, and it is
    # reported in the optimizer domain, so the flip shows there.
    assert np.any(np.abs(minimized.target_objective) > 0.0)
    assert np.allclose(maximized.target_objective, -minimized.target_objective)
