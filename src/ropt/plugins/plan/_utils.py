"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ropt.results import FunctionResults, Results

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


def _check_linear_constraints(
    config: EnOptConfig,
    variables: NDArray[np.float64],
    bounds: NDArray[np.float64],
    tolerance: float,
) -> bool:
    if config.linear_constraints is None:
        return False
    assert config.linear_constraints is not None
    coefficients = config.linear_constraints.coefficients
    constraint_values = np.empty(
        variables.shape[:-1] + (bounds.size,), dtype=np.float64
    )
    for idx in np.ndindex(*variables.shape[:-1]):
        constraint_values[idx] = np.matmul(coefficients, variables[idx])
    return bool(np.any(constraint_values - bounds > tolerance))


def _check_nonlinear_constraints(
    constraints: NDArray[np.float64],
    bounds: NDArray[np.float64],
    tolerance: float,
) -> bool:
    return bool(np.any(constraints - bounds > tolerance))


def _check_constraints(
    config: EnOptConfig, results: FunctionResults, tolerance: float | None
) -> bool:
    if tolerance is None:
        return True

    if _check_bounds(
        -results.evaluations.variables,
        -config.variables.lower_bounds,
        tolerance,
    ) or _check_bounds(
        results.evaluations.variables,
        config.variables.upper_bounds,
        tolerance,
    ):
        return False

    if config.linear_constraints is not None and (
        _check_linear_constraints(
            config,
            -results.evaluations.variables,
            -config.linear_constraints.lower_bounds,
            tolerance,
        )
        or _check_linear_constraints(
            config,
            results.evaluations.variables,
            config.linear_constraints.upper_bounds,
            tolerance,
        )
    ):
        return False

    if config.nonlinear_constraints is not None and results.functions is not None:
        assert results.functions.constraints is not None
        if _check_nonlinear_constraints(
            -results.functions.constraints,
            -config.nonlinear_constraints.lower_bounds,
            tolerance,
        ) or _check_nonlinear_constraints(
            results.functions.constraints,
            config.nonlinear_constraints.upper_bounds,
            tolerance,
        ):
            return False

    return True


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
