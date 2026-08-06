"""Shared helpers for the builtin evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import EvaluationFunctionContext

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from ropt.evaluation import EvaluationBatchContext

    from .base import EvaluationFunctionResult


def _active_evaluations(
    evaluator_context: EvaluationBatchContext,
    batch_id: int,
) -> Iterator[tuple[int, EvaluationFunctionContext]]:
    for eval_idx, realization in enumerate(evaluator_context.realizations):
        if (
            evaluator_context.active is not None
            and not evaluator_context.active[eval_idx]
        ):
            continue
        perturbation = (
            -1
            if evaluator_context.perturbations is None
            else int(evaluator_context.perturbations[eval_idx])
        )
        yield (
            eval_idx,
            EvaluationFunctionContext(
                realization=int(realization),
                perturbation=perturbation,
                batch_id=batch_id,
                eval_idx=eval_idx,
            ),
        )


def _scatter_result(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    eval_idx: int,
    result: EvaluationFunctionResult,
    results: NDArray[np.float64],
    metadata: dict[str, NDArray[Any]],
    objective_count: int,
    eval_count: int,
) -> None:
    results[eval_idx, :objective_count] = result.objectives
    if result.constraints is not None:
        results[eval_idx, objective_count:] = result.constraints
    if result.metadata is not None:
        for key, value in result.metadata.items():
            if key not in metadata:
                metadata[key] = np.zeros(
                    eval_count,
                    dtype=(
                        np.array(value).dtype
                        if isinstance(value, (int, float, complex, np.number))
                        else object
                    ),
                )
            metadata[key][eval_idx] = value
