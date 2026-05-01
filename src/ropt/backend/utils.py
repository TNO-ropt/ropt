"""Utility functions for use by optimizer backend plugins.

This module provides helpers for constraint validation, linear constraint
adjustment based on variable masks, unique output path construction, and
normalization of non-linear constraints into a standard form.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ropt.context import EnOptContext

_MESSAGES = {
    "bounds": "bound constraints",
    "linear:eq": "linear equality constraints",
    "linear:ineq": "linear inequality constraints",
    "nonlinear:eq": "non-linear equality constraints",
    "nonlinear:ineq": "non-linear inequality constraints",
}


def validate_supported_constraints(
    context: EnOptContext,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    """Raise if the context's constraints are incompatible with the chosen method.

    Checks bounds, linear, and non-linear constraints in `context` against the
    sets of method names that support or require each constraint type. Constraint
    types are identified by the keys `"bounds"`, `"linear:eq"`, `"linear:ineq"`,
    `"nonlinear:eq"`, and `"nonlinear:ineq"`.

    Raises `NotImplementedError` if a constraint present in `context` is not
    supported by the method, or if a constraint required by the method is absent
    from `context`.

    Args:
        context:               The optimization context to inspect.
        method:                The name of the optimization method being used.
        supported_constraints: Maps each constraint type to the supported methods.
        required_constraints:  Maps each constraint type to  supported methods.
    """
    _validate_bounds(context, method, supported_constraints, required_constraints)
    _validate_linear_constraints(
        context, method, supported_constraints, required_constraints
    )
    _validate_nonlinear_constraints(
        context, method, supported_constraints, required_constraints
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
    context: EnOptContext,
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
            np.isfinite(context.variables.lower_bounds).any()
            or np.isfinite(context.variables.upper_bounds).any(),
        ),
    )


def _validate_linear_constraints(
    context: EnOptContext,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    if context.linear_constraints is None:
        return

    _check_constraint(
        "linear:ineq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=not bool(
            np.allclose(
                context.linear_constraints.lower_bounds,
                context.linear_constraints.upper_bounds,
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
                context.linear_constraints.lower_bounds,
                context.linear_constraints.upper_bounds,
                rtol=0.0,
                atol=1e-15,
            )
        ),
    )


def _validate_nonlinear_constraints(
    context: EnOptContext,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    nonlinear_constraints = context.nonlinear_constraints
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

    Builds a path from the provided components. If the resulting path already
    exists on disk, a three-digit counter suffix (e.g. `-001`, `-002`) is
    appended or incremented until a non-existing path is found.

    The path is assembled as:
    `<base_dir>/<base_name>[-<name>][-<index>][<suffix>]`

    `base_dir` is created (including parents) if it does not exist.

    Args:
        base_name: Base file or directory name.
        base_dir:  Parent directory. If `None`, the path is relative to the
                   current working directory.
        name:      Optional label appended to `base_name` with a `-` separator.
        suffix:    Optional file extension including the leading dot

    Returns:
        A `pathlib.Path` that does not currently exist on disk.
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
    """Normalizes non-linear constraints into a standard scalar form.

    Transforms raw constraints defined by lower and upper bound pairs into one
    of the forms `C(x) = 0`, `C(x) > 0`, or `C(x) < 0` by subtracting the
    relevant bound and optionally flipping the sign.

    **Normalization rules** (applied per constraint row):

    - If `lower_bound ≈ upper_bound` (within 1e-15): equality constraint,
      normalized as `C(x) - lower_bound`.
    - If only `lower_bound` is finite: inequality, normalized as
      `C(x) - lower_bound`.
    - If only `upper_bound` is finite: inequality, normalized as
      `-(C(x) - upper_bound)`.
    - If both bounds are finite: the constraint is split into two rows, one
      for each bound.

    By default inequality constraints are normalized to `C(x) > 0`. Setting
    `flip=True` produces `C(x) < 0` instead.

    **Usage:**

    1. Initialize with the lower and upper bounds.
    2. Before each new function/gradient evaluation with a new variable vector,
       reset the normalized constraints by calling the `reset` method.
    3. The constraint values are given by the `constraints` property. Before
       accessing it, call the `set_constraints` with the raw constraints. If
       necessary, this will calculate and cache the normalized values. Since
       values are cached, calling this method and accessing `constraints`
       multiple times is cheap.
    4. Use the same procedure for gradients, using the `gradients` property and
       `set_gradients`. Raw gradients must be provided as a matrix, where the
       rows are the gradients of each constraint.
    5. Use the `is_eq` property to retrieve a vector of boolean flags to check
       which constraints are equality constraints.

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
        *,
        flip: bool = False,
    ) -> None:
        """Create a new constraint normalizer.

        Args:
            flip: Whether to normalize inequality constraints to `C(x) < 0`
                  instead of the default `C(x) > 0`.
        """
        self._apply_flip = flip
        self._constraints: NDArray[np.float64] | None = None
        self._gradients: NDArray[np.float64] | None = None
        self._lower_bounds: NDArray[np.float64] | None = None
        self._upper_bounds: NDArray[np.float64] | None = None
        self._is_eq: list[bool] = []
        self._indices: list[int] = []
        self._is_lower: list[bool] = []

    def set_bounds(
        self,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> None:
        """Set or update the constraint bounds.

        Computes the internal normalization mapping from the supplied bound
        arrays. If the bounds change between calls, the mapping is rebuilt.

        Args:
            lower_bounds: Lower bound for each raw constraint.
            upper_bounds: Upper bound for each raw constraint.

        Raises:
            ValueError: If the new bounds change which constraints are
                equalities vs. inequalities, since this would invalidate
                cached optimizer state.
        """
        if (
            self._lower_bounds is None
            or self._upper_bounds is None
            or not np.array_equal(self._lower_bounds, lower_bounds)
            or not np.array_equal(self._upper_bounds, upper_bounds)
        ):
            is_eq = self._is_eq
            self._lower_bounds = lower_bounds.copy()
            self._upper_bounds = upper_bounds.copy()
            self._is_eq = []
            self._indices = []
            self._is_lower = []

            for idx, (lower_bound, upper_bound) in enumerate(
                zip(lower_bounds, upper_bounds, strict=True)
            ):
                if abs(upper_bound - lower_bound) < 1e-15:  # noqa: PLR2004
                    self._is_eq.append(True)
                    self._indices.append(idx)
                    self._is_lower.append(True)
                else:
                    if np.isfinite(lower_bound):
                        self._is_eq.append(False)
                        self._indices.append(idx)
                        self._is_lower.append(True)
                    if np.isfinite(upper_bound):
                        self._is_eq.append(False)
                        self._indices.append(idx)
                        self._is_lower.append(False)

            if is_eq and is_eq != self._is_eq:
                msg = "Some constraints have changed type (equality/inequality)."
                raise ValueError(msg)

    @property
    def is_eq(self) -> list[bool]:
        """Return flags indicating which normalized rows are equalities.

        The returned list corresponds to the normalized constraint rows after
        any splitting of two-sided bounds into separate lower and upper rows.

        Returns:
            A list of booleans where `True` marks a normalized equality
                constraint row.
        """
        return self._is_eq

    def reset(self) -> None:
        """Discard cached normalized values and gradients.

        Call this before normalizing results for a new variable vector. After
        reset, the next calls to `set_constraints` and `set_gradients` will
        rebuild the cached normalized data.

        After calling this method, the `constraints` and `gradients` properties
        return `None` until new values are cached.
        """
        self._constraints = None
        self._gradients = None

    @property
    def constraints(self) -> NDArray[np.float64] | None:
        """Return cached normalized constraint values, if available.

        These values are produced by
        [`set_constraints`][ropt.backend.utils.NormalizedConstraints.set_constraints]
        after subtracting the relevant bound and applying any required sign
        flip.

        Returns `None` if `set_constraints` has not been called since the last
        `reset`.

        Returns:
            A 2D NumPy array of shape `(n_normalized_constraints, n_points)`,
                or `None` when no normalized values are cached.
        """
        return self._constraints

    @property
    def gradients(self) -> NDArray[np.float64] | None:
        """Return cached normalized constraint gradients, if available.

        These gradients are produced by
        [`set_gradients`][ropt.backend.utils.NormalizedConstraints.set_gradients]
        after applying any required sign flip.

        Returns `None` if `set_gradients` has not been called since the last
        `reset`.

        Returns:
            A 2D NumPy array of shape `(n_normalized_constraints, n_variables)`,
                or `None` when no normalized gradients are cached.
        """
        return self._gradients

    def set_constraints(self, values: NDArray[np.float64]) -> None:
        """Normalize and cache raw constraint values.

        Applies the configured bound subtraction and sign convention to raw
        constraint values, then stores the result in the `constraints` cache.

        Parallel evaluation is supported: if `values` is 2D, columns represent
        different evaluation points and rows represent raw constraint indices.
        A 1D input is treated as a single evaluation point.

        If normalized values are already cached, this method returns without
        modifying them; call `reset` first to recompute.

        Args:
            values: Raw constraint values with shape `(n_constraints,)` or
                `(n_constraints, n_points)`.
        """
        if self._constraints is not None:
            return
        if values.ndim == 1:
            values = np.expand_dims(values, axis=1)
        self._constraints = np.empty(
            (len(self._indices), values.shape[1]), dtype=np.float64
        )
        assert self._lower_bounds is not None
        assert self._upper_bounds is not None
        for idx, constraint_idx in enumerate(self._indices):
            rhs_value = (
                self._lower_bounds[constraint_idx]
                if self._is_lower[idx]
                else self._upper_bounds[constraint_idx]
            )
            self._constraints[idx, :] = values[constraint_idx, :] - rhs_value
            flip = self._apply_flip if self._is_lower[idx] else not self._apply_flip
            if flip:
                self._constraints[idx, :] = -self._constraints[idx, :]

    def set_gradients(self, values: NDArray[np.float64]) -> None:
        """Normalize and cache raw constraint gradients.

        Applies the configured sign convention to raw constraint gradients and
        stores the result in the `gradients` cache.

        If normalized gradients are already cached, this method returns without
        modifying them; call `reset` first to recompute.

        Note:
            Unlike `set_constraints`, this method does *not* support parallel
            evaluation; it expects gradients for a single variable vector.

        Args:
            values: Raw constraint gradients with shape
                `(n_constraints, n_variables)`.
        """
        if self._gradients is not None:
            return
        self._gradients = np.empty(
            (len(self._indices), values.shape[1]), dtype=np.float64
        )
        for idx, constraint_idx in enumerate(self._indices):
            flip = self._apply_flip if self._is_lower[idx] else not self._apply_flip
            self._gradients[idx, :] = values[constraint_idx, :]
            if flip:
                self._gradients[idx, :] = -self._gradients[idx, :]


def get_masked_linear_constraints(
    context: EnOptContext, initial_values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Adjust linear constraints based on a variable mask.

    When an optimization problem uses a variable mask (`context.variables.mask`)
    to optimize only a subset of variables, the linear constraints need to be
    adapted. This function performs that adaptation.

    It removes columns from the constraint coefficient matrix
    (`context.linear_constraints.coefficients`) that correspond to the masked
    (fixed) variables. The contribution of these fixed variables (using their
    `initial_values`) is then calculated and subtracted from the original lower
    and upper bounds (`context.linear_constraints.lower_bounds`,
    `context.linear_constraints.upper_bounds`) to produce adjusted bounds for the
    optimization involving only the active variables.

    Additionally, any constraint rows that originally involved *only* masked
    variables (i.e., all coefficients for active variables in that row are zero)
    are removed entirely, as they become trivial constants.

    Args:
        context:        The [`EnOptContext`][ropt.context.EnOptContext] object
                        containing the variable mask and linear constraints.
        initial_values: The initial values to use.

    Returns:
        A tuple of `(coefficients, lower_bounds, upper_bounds)` for the active
            variables only, with the contributions of fixed variables already
            subtracted from the bounds.
    """
    assert context.linear_constraints is not None
    mask = context.variables.mask
    coefficients = context.linear_constraints.coefficients
    lower_bounds = context.linear_constraints.lower_bounds
    upper_bounds = context.linear_constraints.upper_bounds
    if not np.all(mask):
        # Keep rows that only contain non-zero values for the active variables:
        keep_rows = np.all(coefficients[:, ~mask] == 0, axis=1)
        coefficients = coefficients[keep_rows, :]
        lower_bounds = lower_bounds[keep_rows]
        upper_bounds = upper_bounds[keep_rows]
        offsets = np.matmul(coefficients[:, ~mask], initial_values[~mask])
        coefficients = coefficients[:, mask]
    else:
        offsets = 0
    return coefficients, lower_bounds - offsets, upper_bounds - offsets
