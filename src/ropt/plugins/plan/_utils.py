"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ropt.results import FunctionResults, Results


def _get_new_optimal_result(
    optimal_result: Optional[FunctionResults], results: FunctionResults
) -> Optional[FunctionResults]:
    if optimal_result is None:
        return results
    assert optimal_result.functions is not None
    assert results.functions is not None
    optimal = optimal_result.functions.weighted_objective
    objective = results.functions.weighted_objective
    if objective < optimal:
        return results
    return None


def _check_bound_constraints(
    results: FunctionResults, constraint_tolerance: float
) -> bool:
    if results.bound_constraints is not None:
        lower_violations = results.bound_constraints.scaled_lower_violations
        if lower_violations is None:
            lower_violations = results.bound_constraints.lower_violations
        upper_violations = results.bound_constraints.scaled_upper_violations
        if upper_violations is None:
            upper_violations = results.bound_constraints.upper_violations
        for violations in (lower_violations, upper_violations):
            if (
                violations is not None
                and np.any(violations > constraint_tolerance).item()
            ):
                return False
    return True


def _check_constraints(
    results: FunctionResults, constraint_tolerance: Optional[float]
) -> bool:
    if constraint_tolerance is None:
        return True

    if not _check_bound_constraints(results, constraint_tolerance):
        return False

    if results.linear_constraints is not None and (
        results.linear_constraints.violations is not None
        and np.any(results.linear_constraints.violations > constraint_tolerance).item()
    ):
        return False

    if results.nonlinear_constraints is not None:
        violations = results.nonlinear_constraints.scaled_violations
        if violations is None:
            violations = results.nonlinear_constraints.violations
        if violations is not None and np.any(violations > constraint_tolerance).item():
            return False

    return True


def _get_last_result(
    results: Tuple[Results, ...], constraint_tolerance: Optional[float]
) -> Optional[FunctionResults]:
    return next(
        (
            item
            for item in reversed(results)
            if (
                isinstance(item, FunctionResults)
                and item.functions is not None
                and _check_constraints(item, constraint_tolerance)
            )
        ),
        None,
    )


def _update_optimal_result(
    optimal_result: Optional[FunctionResults],
    results: Tuple[Results, ...],
    constraint_tolerance: Optional[float],
) -> Optional[FunctionResults]:
    return_result: Optional[FunctionResults] = None
    for item in results:
        if (
            isinstance(item, FunctionResults)
            and item.functions is not None
            and _check_constraints(item, constraint_tolerance)
        ):
            new_optimal_result = _get_new_optimal_result(optimal_result, item)
            if new_optimal_result is not None:
                optimal_result = new_optimal_result
                return_result = new_optimal_result
    return return_result
