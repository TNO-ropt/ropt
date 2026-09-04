"""Estimation of scales from the first batch of evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.context import EnOptContext
    from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult


def set_auto_scales(
    context: EnOptContext,
    evaluator_context: EvaluationBatchContext,
    evaluator_result: EvaluationBatchResult,
) -> None:
    """Estimate the scales of a run from a batch of evaluations.

    Does nothing unless auto-scaling is configured and the scales are still
    missing, so that only the first batch is used and every later batch is
    scaled the same way.

    Args:
        context:           The context of the run.
        evaluator_context: The context the batch was evaluated with.
        evaluator_result:  The results of the batch, unscaled.
    """
    if not context._needs_auto_scales():  # ruff: ignore[private-member-access]
        return

    # Perturbed values are spread around the point of interest, so the scale is
    # estimated from the unperturbed values. A gradient-only batch has none, and
    # then the perturbed values are all there is to go on.
    rows = (
        np.ones(evaluator_context.realizations.shape, dtype=np.bool_)
        if evaluator_context.perturbations is None
        else evaluator_context.perturbations < 0
    )
    if not rows.any():
        rows = np.ones(evaluator_context.realizations.shape, dtype=np.bool_)
    # Rows that were not evaluated hold zeros rather than results.
    rows &= evaluator_context.active
    realizations = evaluator_context.realizations[rows]
    weights = context.realizations.weights

    objectives = None
    if context.objectives.auto_scale:
        averages = _weighted_average(
            evaluator_result.objectives[rows, :], realizations, weights, "objectives"
        )
        # One factor for all objectives, so that their relative magnitudes, and
        # hence the meaning of their weights, are left alone.
        objectives = _check(
            np.abs(np.dot(averages, context.objectives.weights)), "objectives"
        )

    constraints = None
    if (
        context.nonlinear_constraints is not None
        and context.nonlinear_constraints.auto_scale.any()
    ):
        assert evaluator_result.constraints is not None
        auto_scale = context.nonlinear_constraints.auto_scale
        averages = _weighted_average(
            evaluator_result.constraints[rows, :], realizations, weights, "constraints"
        )
        # A constraint that is not auto-scaled keeps a factor of one, and its
        # estimate is never inspected: it may legitimately be zero.
        constraints = np.where(
            auto_scale, _check(np.abs(averages), "constraints", auto_scale), 1.0
        )

    context._set_auto_scales(objectives, constraints)  # ruff: ignore[private-member-access]


def _weighted_average(
    functions: NDArray[np.float64],
    realizations: NDArray[np.intc],
    weights: NDArray[np.float64],
    what: str,
) -> NDArray[np.float64]:
    weights = np.tile(weights[realizations, np.newaxis], functions.shape[1])
    # A failed realization has nothing to contribute to the estimate.
    failed = np.isnan(functions)
    weights = np.where(failed, 0.0, weights)
    totals = np.sum(weights, axis=0)
    if np.any(np.abs(totals) < np.finfo(np.float64).eps):
        msg = f"Auto-scaling of the {what} found no realizations to average"
        raise RuntimeError(msg)
    return np.sum(np.where(failed, 0.0, functions) * weights / totals, axis=0)


def _check(
    values: NDArray[np.float64],
    what: str,
    selected: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    checked = values if selected is None else values[selected]
    if not np.all(np.isfinite(checked)) or np.any(
        np.abs(checked) < np.finfo(np.float64).eps
    ):
        msg = f"Auto-scaling of the {what} failed to estimate a scale factor"
        raise RuntimeError(msg)
    return values
