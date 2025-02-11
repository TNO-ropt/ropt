"""Utility functions for use by optimizer plugins.

This module provides utility functions to validate supported constraints, filter
linear constraints, and to retrieve the list of supported optimizers.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig

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
    if config.linear_constraints is None:
        return

    _check_constraint(
        "linear:ineq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=not bool(
            np.allclose(
                config.linear_constraints.lower_bounds,
                config.linear_constraints.upper_bounds,
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
                config.linear_constraints.lower_bounds,
                config.linear_constraints.upper_bounds,
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
        have_constraint=not bool(
            np.allclose(
                nonlinear_constraints.lower_bounds,
                nonlinear_constraints.upper_bounds,
                rtol=0.0,
                atol=1e-15,
            )
        ),
    )

    _check_constraint(
        "nonlinear:eq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(
            np.allclose(
                nonlinear_constraints.lower_bounds,
                nonlinear_constraints.upper_bounds,
                rtol=0.0,
                atol=1e-15,
            )
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


class NormalizedConstraints:
    """Class for handling normalized constraints.

    This class can be used to normalize non-linear constraints into the form
    C(x) = 0, C(x) <= 0, or C(x) >= 0. By default this is done by subtracting
    the right-hand side value, and multiplying with -1, if necessary.

    The right hand sides are provided by the `lower_bounds` and `upper_bound`
    values. If corresponding entries in these arrays are equeal (within a 1e-15
    tolerance), the corresponding constraint is assumed to be a equality
    constraint. If they are not, they are considered inequality constraints, if
    one or both values are finite. If the lower bounds are finite, the
    constraint is added as is, after subtracting of the lower bound. If the
    upper bound is finite, the same is done, but the constraint is multiplied by
    -1. If both are finite, both constraints are added, effectively splitting a
    two-sided constraint into two normalized constraints.

    By default this normalizes inequality constraints to the form C(x) < 0, by
    setting `flip` flag, this can be changed to C(x) > 0.

    Usage:
        1. Initialize with the lower and upper bounds.
        2. Before each new function/gradient evaluation with a new variable
           vector, reset the normalized constraints by calling the `reset`
           method.
        3. The constraint values are given by the `constraints` property. Before
           accessing it, call the `set_constraints` with the raw constraints. If
           necessary, this will calculate and cache the normalized values. Since
           values are cached, calling this method and accessing `constraints`
           multiple times is cheap.
        4. Use the same procedure for gradients, using the `gradients` property
           and `set_gradients`. Raw gradients must be provided as a matrix,
           where the rows are the gradients of each constraint.
        5. Use the `is_eq` property to retrieve a vector of boolean flags to
           check which constraints are equality constraints.

        See the `scipy` optimization backend in the `ropt` source code for an
        example of usage.

    Note: Parallel evaluation.
        The raw constraints may be a vector of constraints, or may be a matrix
        of constraints for multiple variables to support parallel evaluation. In
        the latter case, the constraints for different variables are given by
        the columns of the matrix. In this case, the `constraints` property will
        have the same structure. Note that this is only supported for the
        constraint values, not for the gradients. Hence, parallel evaluation of
        multiple gradients is not supported.
    """

    def __init__(
        self,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        *,
        flip: bool = False,
    ) -> None:
        """Initialize the normalization class.

        Args:
            lower_bounds: The lower bounds on the right hand sides.
            upper_bounds: The upper bounds on the right hand sides.
            flip: Whether to flip the sign of the constraints.
        """
        self._apply_flip = flip
        self._is_eq: list[bool] = []
        self._indices: list[int] = []
        self._rhs: list[float] = []
        self._flip: list[bool] = []

        self._constraints: NDArray[np.float64] | None = None
        self._gradients: NDArray[np.float64] | None = None

        for idx, (lower_bound, upper_bound) in enumerate(
            zip(lower_bounds, upper_bounds, strict=True), start=len(self._is_eq)
        ):
            if abs(upper_bound - lower_bound) < 1e-15:  # noqa: PLR2004
                self._is_eq.append(True)
                self._indices.append(idx)
                self._rhs.append(lower_bound)
                self._flip.append(self._apply_flip)
            else:
                if np.isfinite(lower_bound):
                    self._is_eq.append(False)
                    self._indices.append(idx)
                    self._rhs.append(lower_bound)
                    self._flip.append(self._apply_flip)
                if np.isfinite(upper_bound):
                    self._is_eq.append(False)
                    self._indices.append(idx)
                    self._rhs.append(upper_bound)
                    self._flip.append(not self._apply_flip)

    @property
    def is_eq(self) -> list[bool]:
        """Return the flags that indicate equality transforms."""
        return self._is_eq

    def reset(self) -> None:
        """Reset the constraints and its gradients."""
        self._constraints = None
        self._gradients = None

    @property
    def constraints(self) -> NDArray[np.float64] | None:
        """Return the normalized constraints.

        Returns:
            The normalized constraints.
        """
        return self._constraints

    @property
    def gradients(self) -> NDArray[np.float64] | None:
        """Return the normalized constraint gradients.

        Returns:
            The normalized constraint gradients.
        """
        return self._gradients

    def set_constraints(self, values: NDArray[np.float64]) -> None:
        """Set the constraints property.

        Args:
            values: The raw constraint values.
        """
        if values.ndim == 1:
            values = np.expand_dims(values, axis=1)
        self._constraints = np.empty(
            (len(self._indices), values.shape[1]), dtype=np.float64
        )
        for idx, (constraint_idx, rhs_value, flip) in enumerate(
            zip(self._indices, self._rhs, self._flip, strict=True)
        ):
            self._constraints[idx, :] = values[constraint_idx, :] - rhs_value
            if flip:
                self._constraints[idx, :] = -self._constraints[idx, :]

    def set_gradients(self, values: NDArray[np.float64]) -> None:
        """Set the normalized and gradients.

        Args:
            values: The raw gradient values.
        """
        self._gradients = np.empty(
            (len(self._indices), values.shape[1]), dtype=np.float64
        )
        for idx, (constraint_idx, flip) in enumerate(
            zip(self._indices, self._flip, strict=True)
        ):
            self._gradients[idx, :] = values[constraint_idx, :]
            if flip:
                self._gradients[idx, :] = -self._gradients[idx, :]
