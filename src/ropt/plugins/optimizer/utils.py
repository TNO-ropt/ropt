"""Utility functions for use by optimizer plugins.

This module provides utility functions to validate supported constraints, filter
linear constraints, and to retrieve the list of supported optimizers.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig, LinearConstraintsConfig
from ropt.enums import ConstraintType


def filter_linear_constraints(
    config: LinearConstraintsConfig, variable_indices: NDArray[np.intc]
) -> LinearConstraintsConfig:
    """Filter unnecessary constraints from a linear constraint configuration.

    In the case that the optimizer only optimizes a sub-set of the variables,
    linear constraints that are only formed from the unused variables are
    superfluous. This utility function removes those constraints from a  linear
    configuration constraint.

    Args:
        config:           The linear configuration constraint.
        variable_indices: The indices of the variables used by the optimizer.

    Returns:
        The filtered linear constraint configuration.
    """
    # Keep rows that only contain non-zero values for the active variables:
    mask = np.ones(config.coefficients.shape[-1], dtype=np.bool_)
    mask[variable_indices] = False
    keep_rows = np.all(config.coefficients[:, mask] == 0, axis=1)
    coefficients = config.coefficients[keep_rows, :]
    lower_bounds = config.lower_bounds[keep_rows]
    upper_bounds = config.upper_bounds[keep_rows]
    # Keep coefficients for the active variables:
    coefficients = coefficients[:, variable_indices]

    return LinearConstraintsConfig(
        coefficients=coefficients,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


_MESSAGES = {
    "bounds": "bound constraints",
    "linear:eq": "linear equality constraints",
    "linear:ineq": "linear inequality constraints",
    "nonlinear:eq": "non-linear equality constraints",
    "nonlinear:ineq": "non-linear inequality constraints",
}


def validate_supported_constraints(
    config: EnOptConfig,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    """Check if the requested optimization features are supported or required.

    The keys of the supported_constraints and required_constraints dicts specify
    the type of the constraint as shown in the example below. The values are
    sets of method names that support or require the type of constraint
    specified by the key.

    For example:
    {
        "bounds": {"L-BFGS-B", "TNC", "SLSQP"},
        "linear:eq": {"SLSQP"},
        "linear:ineq": {"SLSQP"},
        "nonlinear:eq": {"SLSQP"},
        "nonlinear:ineq": {"SLSQP"},
    }

    Args:
        config:                The ensemble optimizer configuration object.
        method:                The method to check.
        supported_constraints: Specify the supported constraints.
        required_constraints:  Specify the required constraints.
    """
    _validate_bounds(config, method, supported_constraints, required_constraints)
    _validate_linear_constraints(
        config, method, supported_constraints, required_constraints
    )
    _validate_nonlinear_constraints(
        config, method, supported_constraints, required_constraints
    )


def _check_constraint(
    constraint_type: str,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
    *,
    have_constraint: bool,
) -> None:
    supported = {
        algo.lower() for algo in supported_constraints.get(constraint_type, set())
    }
    required = {
        algo.lower() for algo in required_constraints.get(constraint_type, set())
    }
    msg = _MESSAGES[constraint_type]
    if have_constraint and method.lower() not in supported:
        msg = f"optimizer {method} does not support {msg}"
        raise NotImplementedError(msg)
    if not have_constraint and method.lower() in required:
        msg = f"optimizer {method} requires {msg}"
        raise NotImplementedError(msg)


def _validate_bounds(
    config: EnOptConfig,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    _check_constraint(
        "bounds",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(
            np.isfinite(config.variables.lower_bounds).any()
            or np.isfinite(config.variables.upper_bounds).any(),
        ),
    )


def _validate_linear_constraints(
    config: EnOptConfig,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    linear_constraints = config.linear_constraints
    if linear_constraints is None:
        return

    if config.variables.indices is not None:
        linear_constraints = filter_linear_constraints(
            linear_constraints, config.variables.indices
        )

    _check_constraint(
        "linear:ineq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=not bool(
            np.allclose(
                linear_constraints.lower_bounds,
                linear_constraints.upper_bounds,
                rtol=0.0,
                atol=1e-15,
            )
        ),
    )

    _check_constraint(
        "linear:eq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(
            np.allclose(
                linear_constraints.lower_bounds,
                linear_constraints.upper_bounds,
                rtol=0.0,
                atol=1e-15,
            )
        ),
    )


def _validate_nonlinear_constraints(
    config: EnOptConfig,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    nonlinear_constraints = config.nonlinear_constraints
    if nonlinear_constraints is None:
        return

    _check_constraint(
        "nonlinear:ineq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(
            np.any(nonlinear_constraints.types == ConstraintType.LE)
            or np.any(nonlinear_constraints.types == ConstraintType.GE),
        ),
    )

    _check_constraint(
        "nonlinear:eq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(
            np.any(nonlinear_constraints.types == ConstraintType.EQ),
        ),
    )


def create_output_path(
    base_name: str,
    base_dir: Path | None = None,
    name: str | None = None,
    suffix: str | None = None,
) -> Path:
    """Create an output path name.

    If the path already exists, an index is appended to it.

    Args:
        base_name: Base name of the path.
        base_dir:  Optional directory to base the path in.
        name:      Optional optimization step name to include in the name.
        suffix:    Optional suffix for the resulting path.

    Returns:
        The constructed path
    """
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
    if name is not None:
        base_name += f"-{name}"
    output = base_dir / base_name if base_dir is not None else Path(base_name)
    if suffix is not None:
        output = output.with_suffix(suffix)
    while output.exists():
        fields = base_name.split("-")
        if fields[-1].strip().isdigit():
            index = int(fields[-1]) + 1
            base_name = "-".join(fields[:-1]) + f"-{index:03}"
        else:
            base_name = f"{base_name}-001"
        output = base_dir / base_name if base_dir is not None else Path(base_name)
        if suffix is not None:
            output = output.with_suffix(suffix)
    return output
