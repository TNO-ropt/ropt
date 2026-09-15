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
    # Yields only the rows that must be evaluated. Inactive rows keep their
    # index, so results still scatter back to the row they came from.
    for eval_idx, realization in enumerate(evaluator_context.realizations):
        if (
            evaluator_context.active is not None
            and not evaluator_context.active[eval_idx]
        ):
            continue
        # An unperturbed evaluation is marked with -1, so the function can tell
        # it apart from perturbation 0.
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
                metadata=evaluator_context.metadata,
            ),
        )


def _scatter_result(
    eval_idx: int,
    result: EvaluationFunctionResult,
    results: NDArray[np.float64],
    metadata: dict[str, dict[int, Any]],
    objective_count: int,
) -> None:
    results[eval_idx, :objective_count] = result.objectives
    if result.constraints is not None:
        results[eval_idx, objective_count:] = result.constraints
    if result.metadata is not None:
        for key, value in result.metadata.items():
            metadata.setdefault(key, {})[eval_idx] = value


def _build_metadata(
    metadata: dict[str, dict[int, Any]], eval_count: int
) -> dict[str, NDArray[Any]]:
    return {
        key: _build_metadata_column(key, values, eval_count)
        for key, values in metadata.items()
    }


def _build_metadata_column(
    key: str, values: dict[int, Any], eval_count: int
) -> NDArray[Any]:
    # Rows that never set the key get a missing marker rather than a zero, which
    # would be indistinguishable from a value the evaluator actually returned.
    # Numpy has no integer NaN, so a numeric column that has holes can only keep
    # its values by widening to float.
    strings = sum(isinstance(value, str) for value in values.values())
    if strings and strings != len(values):
        msg = f"Metadata has inconsistent types: {key}"
        raise ValueError(msg)
    numeric = all(
        isinstance(value, (bool, int, float, complex, np.number))
        for value in values.values()
    )
    if numeric and len(values) == eval_count:
        return np.array([values[idx] for idx in range(eval_count)])
    column: NDArray[Any] = (
        np.full(eval_count, np.nan)
        if numeric
        else np.full(eval_count, None, dtype=object)
    )
    for idx, value in values.items():
        column[idx] = value
    return column
