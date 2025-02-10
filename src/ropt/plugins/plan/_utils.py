"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ropt.enums import ConstraintType
from ropt.results import FunctionResults, Functions, Results

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


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


def _check_bounds(
    variables: NDArray[np.float64], bounds: NDArray[np.float64], tolerance: float
) -> bool:
    return bool(
        np.any(np.logical_and(np.isfinite(bounds), variables - bounds > tolerance))
    )


def _get_constraint_violations(
    constraint_values: NDArray[np.float64], types: NDArray[np.ubyte], tolerance: float
) -> bool:
    le_inx = types == ConstraintType.LE
    ge_inx = types == ConstraintType.GE
    constraint_values = constraint_values.copy()
    constraint_values[le_inx] = np.maximum(constraint_values[le_inx], 0.0)
    constraint_values[ge_inx] = np.minimum(constraint_values[ge_inx], 0.0)
    return bool(np.any(np.abs(constraint_values) > tolerance))


def _check_linear_constraints(
    config: EnOptConfig, variables: NDArray[np.float64], tolerance: float
) -> bool:
    if config.linear_constraints is None:
        return False
    assert config.linear_constraints is not None
    coefficients = config.linear_constraints.coefficients
    rhs_values = config.linear_constraints.rhs_values
    constraint_values = np.empty(
        variables.shape[:-1] + (rhs_values.size,), dtype=np.float64
    )
    for idx in np.ndindex(*variables.shape[:-1]):
        constraint_values[idx] = np.matmul(coefficients, variables[idx])
    return _get_constraint_violations(
        constraint_values - rhs_values, config.linear_constraints.types, tolerance
    )


def _check_nonlinear_constraints(
    config: EnOptConfig, functions: Functions, tolerance: float
) -> bool:
    assert config.nonlinear_constraints is not None
    rhs_values = config.nonlinear_constraints.rhs_values
    assert functions.constraints is not None
    return _get_constraint_violations(
        functions.constraints - rhs_values,
        config.nonlinear_constraints.types,
        tolerance,
    )


def _check_constraints(
    config: EnOptConfig, results: FunctionResults, tolerance: float | None
) -> bool:
    if tolerance is None:
        return True

    if _check_bounds(
        -results.evaluations.variables, -config.variables.lower_bounds, tolerance
    ) or _check_bounds(
        results.evaluations.variables, config.variables.upper_bounds, tolerance
    ):
        return False

    if config.linear_constraints is not None and _check_linear_constraints(
        config, results.evaluations.variables, tolerance
    ):
        return False

    return (
        config.nonlinear_constraints is None
        or results.functions is None
        or not _check_nonlinear_constraints(config, results.functions, tolerance)
    )


def _get_last_result(
    config: EnOptConfig,
    results: tuple[Results, ...],
    transformed_results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> FunctionResults | None:
    return next(
        (
            cast(FunctionResults, transformed_item)
            for item, transformed_item in zip(
                reversed(results), reversed(transformed_results), strict=False
            )
            if (
                isinstance(item, FunctionResults)
                and item.functions is not None
                and _check_constraints(config, item, constraint_tolerance)
            )
        ),
        None,
    )


def _update_optimal_result(
    config: EnOptConfig,
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
            and _check_constraints(config, transformed_item, constraint_tolerance)
        ):
            assert isinstance(item, FunctionResults)
            new_optimal_result = _get_new_optimal_result(optimal_result, item)
            if new_optimal_result is not None:
                optimal_result = new_optimal_result
                return_result = new_optimal_result
    return return_result


def _get_all_results(
    config: EnOptConfig,
    results: tuple[Results, ...],
    transformed_results: tuple[Results, ...],
    constraint_tolerance: float | None,
) -> tuple[FunctionResults, ...]:
    return tuple(
        cast(FunctionResults, item)
        for item, transformed_item in zip(results, transformed_results, strict=False)
        if (
            isinstance(transformed_item, FunctionResults)
            and transformed_item.functions is not None
            and _check_constraints(config, transformed_item, constraint_tolerance)
        )
    )
