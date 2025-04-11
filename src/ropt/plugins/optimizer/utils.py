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
    """Validate if the configured constraints are supported by the chosen method.

    This function checks if the constraints defined in the `config` object
    (bounds, linear, non-linear) are compatible with the specified optimization
    `method`. It uses dictionaries mapping constraint types to sets of methods
    that support or require them.

    Constraint types are identified by keys like `"bounds"`, `"linear:eq"`,
    `"linear:ineq"`, `"nonlinear:eq"`, and `"nonlinear:ineq"`.

    Example `supported_constraints` dictionary:
    ```python
    {
        "bounds": {"L-BFGS-B", "TNC", "SLSQP"},
        "linear:eq": {"SLSQP"},
        "linear:ineq": {"SLSQP"},
        "nonlinear:eq": {"SLSQP"},
        "nonlinear:ineq": {"SLSQP"},
    }
    ```
    A similar structure is used for `required_constraints`.

    Args:
        config:                The optimization configuration object.
        method:                The name of the optimization method being used.
        supported_constraints: Dict mapping constraint types to sets of methods
                               that support them.
        required_constraints:  Dict mapping constraint types to sets of methods
                               that require them.

    Raises:
        NotImplementedError: If a configured constraint is not supported by the
                             method, or if a required constraint is missing.
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
    """Construct a unique output path, appending an index if necessary.

    This function generates a file or directory path based on the provided
    components. If the resulting path already exists, it automatically appends
    or increments a numerical suffix (e.g., "-001", "-002") to ensure uniqueness.

    Args:
        base_name: The core name for the path.
        base_dir:  Optional parent directory for the path.
        name:      Optional identifier (e.g., step name) to include in the path.
        suffix:    Optional file extension or suffix for the path.

    Returns:
        A unique `pathlib.Path` object.
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
    setting `flip` flag, this can be changed to C(x) < 0.

    **Usage:**

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
            flip:         Whether to flip the sign of the constraints.
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
        """Return flags indicating which constraints are equality constraints.

        Returns:
            A list of booleans, `True` for constraints that are equality constraints.
        """
        return self._is_eq

    def reset(self) -> None:
        """Reset cached normalized constraints and gradients.

        This should be called before evaluating with a new variable vector to
        ensure fresh values are calculated upon the next access.
        """
        self._constraints = None
        self._gradients = None

    @property
    def constraints(self) -> NDArray[np.float64] | None:
        """Return the cached normalized constraint values.

        These are the constraint values after applying the normalization logic
        (subtracting RHS, potential sign flipping) based on the bounds provided
        during initialization.

        This property should be accessed after calling
        [`set_constraints`][ropt.plugins.optimizer.utils.NormalizedConstraints.set_constraints]
        with the raw constraint values for the current variable vector. Returns
        `None` if `set_constraints` has not been called since the last `reset`.

        Returns:
            A NumPy array containing the normalized constraint values.
        """
        return self._constraints

    @property
    def gradients(self) -> NDArray[np.float64] | None:
        """Return the cached normalized constraint gradients.

        These are the gradients of the constraints after applying the
        normalization logic (potential sign flipping) based on the bounds
        provided during initialization.

        This property should be accessed after calling
        [`set_gradients`][ropt.plugins.optimizer.utils.NormalizedConstraints.set_gradients]
        with the raw constraint gradients for the current variable vector.
        Returns `None` if `set_gradients` has not been called since the last
        `reset`.

        Returns:
            A 2D NumPy array containing the normalized constraint gradients.
        """
        return self._gradients

    def set_constraints(self, values: NDArray[np.float64]) -> None:
        """Calculate and cache normalized constraint values.

        This method takes the raw constraint values (evaluated at the current
        variable vector) and applies the normalization logic defined during
        initialization (subtracting RHS, potential sign flipping). The results
        are stored internally and made available via the `constraints` property.

        This supports parallel evaluation: if `values` is a 2D array, each row
        is treated as the constraint values for a separate variable vector
        evaluation.

        Args:
            values: A 1D or 2D NumPy array of raw constraint values. If 2D,
                    rows represent different evaluations.
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
        """Calculate and cache normalized constraint gradients.

        This method takes the raw constraint gradients (evaluated at the current
        variable vector) and applies the normalization logic defined during
        initialization (potential sign flipping). The results are stored
        internally and made available via the `gradients` property.

        Note:
            Unlike `set_constraints`, this method does *not* support parallel
            evaluation; it expects gradients corresponding to a single variable vector.

        Args:
            values: A 2D NumPy array of raw constraint gradients (rows are
                    gradients of original constraints, columns are variables).
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


def get_masked_linear_constraints(
    config: EnOptConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Adjust linear constraints based on a variable mask.

    When an optimization problem uses a variable mask (`config.variables.mask`)
    to optimize only a subset of variables, the linear constraints need to be
    adapted. This function performs that adaptation.

    It removes columns from the constraint coefficient matrix
    (`config.linear_constraints.coefficients`) that correspond to the masked
    (fixed) variables. The contribution of these fixed variables (using their
    `initial_values`) is then calculated and subtracted from the original lower
    and upper bounds (`config.linear_constraints.lower_bounds`,
    `config.linear_constraints.upper_bounds`) to produce adjusted bounds for the
    optimization involving only the active variables.

    Additionally, any constraint rows that originally involved *only* masked
    variables (i.e., all coefficients for active variables in that row are zero)
    are removed entirely, as they become trivial constants.

    Args:
        config: The [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object
                containing the variable mask and linear constraints.

    Returns:
        The adjusted coefficients and bounds.
    """
    assert config.linear_constraints is not None
    mask = config.variables.mask
    coefficients = config.linear_constraints.coefficients
    lower_bounds = config.linear_constraints.lower_bounds
    upper_bounds = config.linear_constraints.upper_bounds
    if mask is not None:
        # Keep rows that only contain non-zero values for the active variables:
        keep_rows = np.all(coefficients[:, ~mask] == 0, axis=1)
        coefficients = coefficients[keep_rows, :]
        lower_bounds = lower_bounds[keep_rows]
        upper_bounds = upper_bounds[keep_rows]
        offsets = np.matmul(
            coefficients[:, ~mask], config.variables.initial_values[~mask]
        )
        coefficients = coefficients[:, mask]
    else:
        offsets = 0
    return coefficients, lower_bounds - offsets, upper_bounds - offsets
