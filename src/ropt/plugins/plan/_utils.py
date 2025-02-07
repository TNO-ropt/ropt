"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from typing import cast

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


def _check_bound_constraints(
    results: FunctionResults, constraint_tolerance: float
) -> bool:
    if results.bound_constraints is not None:
        for violations in (
            results.bound_constraints.lower_violations,
            results.bound_constraints.upper_violations,
        ):
            if (
                violations is not None
                and np.any(violations > constraint_tolerance).item()
            ):
                return False
    return True


def _check_constraints(
    results: FunctionResults, constraint_tolerance: float | None
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
    results: tuple[Results, ...],
    transformed_results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> FunctionResults | None:
    return next(
        (
            cast(FunctionResults, unscaled_item)
            for unscaled_item, scaled_item in zip(
                reversed(results), reversed(transformed_results), strict=False
            )
            if (
                isinstance(scaled_item, FunctionResults)
                and scaled_item.functions is not None
                and _check_constraints(scaled_item, constraint_tolerance)
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
    for unscaled_item, scaled_item in zip(results, transformed_results, strict=False):
        if (
            isinstance(scaled_item, FunctionResults)
            and scaled_item.functions is not None
            and _check_constraints(scaled_item, constraint_tolerance)
        ):
            assert isinstance(unscaled_item, FunctionResults)
            new_optimal_result = _get_new_optimal_result(optimal_result, unscaled_item)
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
        cast(FunctionResults, unscaled_item)
        for unscaled_item, scaled_item in zip(
            results, transformed_results, strict=False
        )
        if (
            isinstance(scaled_item, FunctionResults)
            and scaled_item.functions is not None
            and _check_constraints(scaled_item, constraint_tolerance)
        )
    )
