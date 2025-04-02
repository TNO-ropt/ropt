"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

import numpy as np

from ropt.results import FunctionResults, Results


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
    if results.constraint_info is None:
        return False

    for violations in (
        results.constraint_info.bound_violation,
        results.constraint_info.linear_violation,
        results.constraint_info.nonlinear_violation,
    ):
        if violations is not None and np.any(violations > tolerance):
            return True

    return False


def _get_last_result(
    results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> FunctionResults | None:
    return next(
        (
            item
            for item in reversed(results)
            if (
                isinstance(item, FunctionResults)
                and item.functions is not None
                and not _violates_constraint(item, constraint_tolerance)
            )
        ),
        None,
    )


def _update_optimal_result(
    optimal_result: FunctionResults | None,
    results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> FunctionResults | None:
    return_result: FunctionResults | None = None
    for item in results:
        if (
            isinstance(item, FunctionResults)
            and item.functions is not None
            and not _violates_constraint(item, constraint_tolerance)
        ):
            assert isinstance(item, FunctionResults)
            new_optimal_result = _get_new_optimal_result(optimal_result, item)
            if new_optimal_result is not None:
                optimal_result = new_optimal_result
                return_result = new_optimal_result
    return return_result
