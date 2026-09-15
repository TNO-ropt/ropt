"""The optimization problem that is handed to a backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from ropt._utils import split_constraints
from ropt.exceptions import UnsupportedError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.context import EnOptContext

_MESSAGES: Final = {
    "bounds": "bound constraints",
    "linear:eq": "linear equality constraints",
    "linear:ineq": "linear inequality constraints",
    "nonlinear:eq": "non-linear equality constraints",
    "nonlinear:ineq": "non-linear inequality constraints",
}


class OptimizationProblem:
    """The problem a backend is asked to solve.

    Everything on this object is **scaled** and lives in **free-variable
    space**: the variables that `variables.mask` fixes are gone, the bounds and
    the linear constraints are reduced to the ones that remain, and the values
    delivered through the [`OptimizerCallback`][ropt.core.OptimizerCallback]
    use the same space. A backend therefore neither scales nor masks anything.

    See [`Backend`][ropt.backend.Backend] for the rest of the contract.

    Attributes:
        initial_values: The starting point, shape `(variable_count,)`.
        lower_bounds:   The lower bounds, shape `(variable_count,)`.
        upper_bounds:   The upper bounds, shape `(variable_count,)`.
        variable_types: The type of each variable, shape `(variable_count,)`.
    """

    def __init__(
        self, context: EnOptContext, initial_values: NDArray[np.float64]
    ) -> None:
        """Reduce a context and a full-space starting point to a problem.

        Args:
            context:        The optimization context.
            initial_values: The values of all variables, including fixed ones.
        """
        mask = context.variables.mask
        self.initial_values = initial_values[mask]
        self.lower_bounds = context.variables.lower_bounds[mask]
        self.upper_bounds = context.variables.upper_bounds[mask]
        self.variable_types = context.variables.types[mask]

        self._linear_constraints = (
            None
            if context.linear_constraints is None
            else _reduce_linear_constraints(context, initial_values)
        )
        self._nonlinear_equalities = _nonlinear_equalities(context)

        # Read from the reduced bounds: a bound on a fixed variable does not
        # constrain the problem the optimizer solves.
        self._have = {
            "bounds": bool(
                np.isfinite(self.lower_bounds).any()
                or np.isfinite(self.upper_bounds).any()
            )
        }
        # The remaining kinds follow the configuration rather than the reduced
        # problem, so that an unsupported constraint is still reported when
        # every row happens to drop out.
        self._have |= _configured_constraints(context)

    @property
    def variable_count(self) -> int:
        """The number of free variables.

        Returns:
            The number of variables the optimizer sees.
        """
        return int(self.initial_values.size)

    @property
    def linear_constraints(
        self,
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.bool_],
        ]
        | None
    ):
        """The linear constraints, reduced to the problem the optimizer solves.

        Rows that cannot constrain the optimization are dropped: a row is kept
        only if it has a non-zero coefficient on a free variable and at least
        one finite bound. The columns of the fixed variables are removed, and
        the contribution those variables make is folded into the bounds.

        The four arrays are aligned, with one entry per surviving constraint.
        `equality` marks the constraints whose bounds coincide; it follows the
        bounds as configured, so scaling cannot change it. Pass the tuple to
        [`split_linear_constraints`][ropt.backend.utils.split_linear_constraints]
        to get the same normalized form as the non-linear constraints.

        Returns:
            `(coefficients, lower_bounds, upper_bounds, equality)`, or `None`.
        """
        return self._linear_constraints

    @property
    def nonlinear_equalities(self) -> NDArray[np.bool_] | None:
        """Which non-linear constraint values are equalities.

        One flag per value delivered through the
        [`OptimizerCallback`][ropt.core.OptimizerCallback]: a constraint with a
        finite lower and a finite upper bound delivers two values, an equality
        one, and a constraint with no finite bound none.

        Returns:
            A flag per value, or `None` if there are no non-linear constraints.
        """
        return self._nonlinear_equalities

    def validate_supported_constraints(
        self,
        method: str,
        supported_constraints: dict[str, set[str]],
        required_constraints: dict[str, set[str]],
    ) -> None:
        """Raise if this problem's constraints do not suit the chosen method.

        Constraint types are identified by the keys `"bounds"`, `"linear:eq"`,
        `"linear:ineq"`, `"nonlinear:eq"` and `"nonlinear:ineq"`.

        Args:
            method:                The name of the optimization method used.
            supported_constraints: Maps each constraint type to the methods
                                   that support it.
            required_constraints:  Maps each constraint type to the methods
                                   that require it.

        Raises:
            UnsupportedError: If a constraint present in the problem is not
                              supported by the method, or a constraint the
                              method requires is absent.
        """
        for constraint_type, have_constraint in self._have.items():
            supported = {
                item.lower()
                for item in supported_constraints.get(constraint_type, set())
            }
            required = {
                item.lower()
                for item in required_constraints.get(constraint_type, set())
            }
            message = _MESSAGES[constraint_type]
            if have_constraint and method.lower() not in supported:
                msg = f"Optimizer '{method}' does not support {message}."
                raise UnsupportedError(msg)
            if not have_constraint and method.lower() in required:
                msg = f"Optimizer '{method}' requires {message}."
                raise UnsupportedError(msg)


def _configured_constraints(context: EnOptContext) -> dict[str, bool]:
    linear = context._linear_equality  # ruff: ignore[private-member-access]
    nonlinear = context._nonlinear_equality  # ruff: ignore[private-member-access]
    return {
        "linear:ineq": linear is not None and bool((~linear).any()),
        "linear:eq": linear is not None and bool(linear.all()),
        "nonlinear:ineq": nonlinear is not None and bool((~nonlinear).any()),
        "nonlinear:eq": nonlinear is not None and bool(nonlinear.all()),
    }


def _nonlinear_equalities(context: EnOptContext) -> NDArray[np.bool_] | None:
    if context.nonlinear_constraints is None:
        return None
    equality = context._nonlinear_equality  # ruff: ignore[private-member-access]
    assert equality is not None
    constraint_index, _ = split_constraints(
        context.nonlinear_constraints.lower_bounds,
        context.nonlinear_constraints.upper_bounds,
        equality,
    )
    return equality[constraint_index]


def _reduce_linear_constraints(
    context: EnOptContext, initial_values: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    assert context.linear_constraints is not None
    equality = context._linear_equality  # ruff: ignore[private-member-access]
    assert equality is not None
    mask = context.variables.mask
    coefficients = context.linear_constraints.coefficients
    lower_bounds = context.linear_constraints.lower_bounds
    upper_bounds = context.linear_constraints.upper_bounds

    keep = np.any(coefficients[:, mask] != 0, axis=1) & (
        np.isfinite(lower_bounds) | np.isfinite(upper_bounds)
    )
    coefficients = coefficients[keep, :]
    offsets = np.matmul(coefficients[:, ~mask], initial_values[~mask])
    return (
        coefficients[:, mask],
        lower_bounds[keep] - offsets,
        upper_bounds[keep] - offsets,
        equality[keep],
    )
