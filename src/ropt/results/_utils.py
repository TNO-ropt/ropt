from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.enums import ConstraintType, ResultAxis


def _get_lower_bound_constraint_values(
    config: EnOptConfig, variables: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    variables = np.moveaxis(variables, axis, -1)
    bounds = config.variables.lower_bounds
    result = np.where(np.isfinite(bounds), variables - bounds, 0.0)
    return np.moveaxis(result, -1, axis)


def _get_lower_bound_violations(
    constraints: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.maximum(-constraints, 0.0)


def _get_upper_bound_constraint_values(
    config: EnOptConfig, variables: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    variables = np.moveaxis(variables, axis, -1)
    bounds = config.variables.upper_bounds
    result = np.where(np.isfinite(bounds), variables - bounds, 0.0)
    return np.moveaxis(result, -1, axis)


def _get_upper_bound_violations(
    constraints: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.maximum(constraints, 0.0)


def _get_linear_constraint_values(
    config: EnOptConfig, variables: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    assert config.linear_constraints is not None
    coefficients = config.linear_constraints.coefficients
    rhs_values = config.linear_constraints.rhs_values
    variables = np.moveaxis(variables, axis, -1)
    constraint_values = np.empty(
        variables.shape[:-1] + (rhs_values.size,), dtype=np.float64
    )
    for idx in np.ndindex(*variables.shape[:-1]):
        constraint_values[idx] = np.matmul(coefficients, variables[idx])
    result = np.array(constraint_values - rhs_values)
    return np.moveaxis(result, -1, axis)


def _get_linear_constraint_violations(
    config: EnOptConfig, constraints: NDArray[np.float64]
) -> NDArray[np.float64]:
    assert config.linear_constraints is not None
    return _get_constraint_violations(constraints, config.linear_constraints.types)


def _get_nonlinear_constraint_values(
    config: EnOptConfig, constraints: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    constraints = np.moveaxis(constraints, axis, -1)
    assert config.nonlinear_constraints is not None
    rhs_values = config.nonlinear_constraints.rhs_values
    assert constraints is not None
    result = constraints - rhs_values
    return np.moveaxis(result, -1, axis)


def _get_nonlinear_constraint_violations(
    config: EnOptConfig, constraints: NDArray[np.float64]
) -> NDArray[np.float64]:
    assert config.nonlinear_constraints is not None
    return _get_constraint_violations(constraints, config.nonlinear_constraints.types)


def _get_constraint_violations(
    constraint_values: NDArray[np.float64], types: NDArray[np.ubyte]
) -> NDArray[np.float64]:
    le_inx = types == ConstraintType.LE
    ge_inx = types == ConstraintType.GE
    constraint_values = constraint_values.copy()
    constraint_values[le_inx] = np.maximum(constraint_values[le_inx], 0.0)
    constraint_values[ge_inx] = np.minimum(constraint_values[ge_inx], 0.0)
    return np.abs(constraint_values)


@overload
def _immutable_copy(data: NDArray[Any]) -> NDArray[Any]: ...


@overload
def _immutable_copy(data: NDArray[Any] | None) -> NDArray[Any] | None: ...


def _immutable_copy(data: NDArray[Any] | None) -> NDArray[Any] | None:
    if data is not None:
        data = data.copy()
        data.setflags(write=False)
    return data


def _get_axis_names(config: EnOptConfig, axis: ResultAxis) -> tuple[str, ...] | None:
    match axis:
        case ResultAxis.VARIABLE:
            return config.variables.get_formatted_names()
        case ResultAxis.OBJECTIVE:
            return config.objectives.names
        case ResultAxis.NONLINEAR_CONSTRAINT:
            assert config.nonlinear_constraints is not None
            return config.nonlinear_constraints.names
        case ResultAxis.REALIZATION:
            return config.realizations.names
    return None
