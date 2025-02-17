"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt.results import FunctionResults, Results

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _get_new_optimal_result(
    optimal_result: FunctionResults | None, results: FunctionResults
) -> FunctionResults | None:
    if optimal_result is None:
        return results
    assert optimal_result.functions is not None
    assert results.functions is not None
    optimal = optimal_result.functions.weighted_objective
    objective = results.functions.weighted_objective
    if objective < optimal:
        return results
    return None


def _violates_constraint(results: Results, tolerance: float | None) -> bool:
    if tolerance is None:
        return False

    assert isinstance(results, FunctionResults)
    if results.constraint_diffs is None:
        return False

    def _check(diffs: NDArray[np.float64] | None, *, flip: bool) -> bool:
        if diffs is None:
            return False
        return bool(np.any((-diffs if flip else diffs) > tolerance))

    return (
        _check(results.constraint_diffs.bound_lower, flip=True)
        or _check(results.constraint_diffs.bound_upper, flip=False)
        or _check(results.constraint_diffs.linear_lower, flip=True)
        or _check(results.constraint_diffs.linear_upper, flip=False)
        or _check(results.constraint_diffs.nonlinear_lower, flip=True)
        or _check(results.constraint_diffs.nonlinear_upper, flip=False)
    )


def _get_last_result(
    results: tuple[Results, ...],
    transformed_results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> FunctionResults | None:
    return next(
        (
            item
            for item, transformed_item in zip(
                reversed(results), reversed(transformed_results), strict=False
            )
            if (
                isinstance(item, FunctionResults)
                and item.functions is not None
                and not _violates_constraint(transformed_item, constraint_tolerance)
            )
        ),
        None,
    )


def _update_optimal_result(
    optimal_result: FunctionResults | None,
    results: tuple[Results, ...],
    transformed_results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> FunctionResults | None:
    return_result: FunctionResults | None = None
    for item, transformed_item in zip(results, transformed_results, strict=False):
        if (
            isinstance(transformed_item, FunctionResults)
            and transformed_item.functions is not None
            and not _violates_constraint(transformed_item, constraint_tolerance)
        ):
            assert isinstance(item, FunctionResults)
            new_optimal_result = _get_new_optimal_result(optimal_result, item)
            if new_optimal_result is not None:
                optimal_result = new_optimal_result
                return_result = new_optimal_result
    return return_result


def _get_all_results(
    results: tuple[Results, ...],
    transformed_results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> tuple[FunctionResults, ...]:
    return tuple(
        item
        for item, transformed_item in zip(results, transformed_results, strict=False)
        if (
            isinstance(item, FunctionResults)
            and item.functions is not None
            and not _violates_constraint(transformed_item, constraint_tolerance)
        )
    )
