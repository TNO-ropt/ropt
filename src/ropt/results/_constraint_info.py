from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.context import EnOptContext


@dataclass(slots=True)
class ConstraintInfo(ResultField):
    """Store information about constraint differences and violations.

    The `ConstraintInfo` class stores the differences between variable or
    constraint values and their respective bounds. It also calculates and stores
    constraint violations. This information is useful for assessing how well the
    optimization process is satisfying the imposed constraints.


    **Constraint differences**

    These represent the difference between a variable or constraint value and
    its corresponding bound. Whether this difference signifies a violation
    depends on the bound type:

    - _Lower Bounds:_ A negative difference means the value is below the lower
      bound, thus violating the constraint.
    - _Upper Bounds:_ A positive difference means the value is above the upper
      bound, thus violating the constraint.

    The class stores the following information on the differences:


    **Constraint Violations**

    Constraint violations are calculated based on the constraint differences. If
    a bound is violated, the violation value is the absolute value of the
    difference. If the bound is not violated, the violation value is zero.


    **Result descriptions**

    The class stores the following information for bound, linear constraint, and
    non-linear constraint differences and violations as one-dimensional vectors:

    === "Bound Constraints"

        - Differences: `bound_lower` and `bound_upper`
        - Violations: `bound_violation`
        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Linear Constraints"

        - Differences: `linear_lower` and `linear_upper`
        - Violations: `linear_violation`
        - Shape: $(n_l,)$, where:
            - $n_l$ is the number of linear constraints.
        - Axis type:
            - [`AxisName.LINEAR_CONSTRAINT`][ropt.enums.AxisName.LINEAR_CONSTRAINT]

    === "Nonlinear Constraints"

        - Differences: `nonlinear_lower` and `nonlinear_upper`
        - Violations: `nonlinear_violation`
        - Shape: $(n_c,)$, where:
            - $n_c$ is the number of non-linear constraints.
        - Axis type:
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

    Attributes:
         bound_lower:         Difference between variables and their lower bounds.
         bound_upper:         Difference between variables and their upper bounds.
         linear_lower:        Difference between linear constraints and their lower
                              bounds.
         linear_upper:        Difference between linear constraints and their upper
                              bounds.
         nonlinear_lower:     Difference between nonlinear constraints and their
                              lower bounds.
         nonlinear_upper:     Difference between nonlinear constraints and their
                              upper bounds.
         bound_violation:     Magnitude of the violation of the variable bounds.
         linear_violation:    Magnitude of the violation of the linear constraints.
         nonlinear_violation: Magnitude of the violation of the nonlinear constraints.
    """

    bound_lower: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.VARIABLE,)}
    )
    bound_upper: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.VARIABLE,)}
    )
    linear_lower: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.LINEAR_CONSTRAINT,)}
    )
    linear_upper: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.LINEAR_CONSTRAINT,)}
    )
    nonlinear_lower: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.NONLINEAR_CONSTRAINT,)}
    )
    nonlinear_upper: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.NONLINEAR_CONSTRAINT,)}
    )
    bound_violation: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.VARIABLE,)}
    )
    linear_violation: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.LINEAR_CONSTRAINT,)}
    )
    nonlinear_violation: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (AxisName.NONLINEAR_CONSTRAINT,)}
    )

    def __post_init__(self) -> None:
        self.bound_lower = _immutable_copy(self.bound_lower)
        self.bound_upper = _immutable_copy(self.bound_upper)
        self.linear_lower = _immutable_copy(self.linear_lower)
        self.linear_upper = _immutable_copy(self.linear_upper)
        self.nonlinear_lower = _immutable_copy(self.nonlinear_lower)
        self.nonlinear_upper = _immutable_copy(self.nonlinear_upper)
        if self.bound_lower is not None and self.bound_upper is not None:
            self.bound_violation = _immutable_copy(
                np.maximum(
                    np.where(self.bound_lower < 0.0, -self.bound_lower, 0.0),
                    np.where(self.bound_upper > 0.0, self.bound_upper, 0.0),
                )
            )
        if self.linear_lower is not None and self.linear_upper is not None:
            self.linear_violation = _immutable_copy(
                np.maximum(
                    np.where(self.linear_lower < 0.0, -self.linear_lower, 0.0),
                    np.where(self.linear_upper > 0.0, self.linear_upper, 0.0),
                )
            )
        if self.nonlinear_lower is not None and self.nonlinear_upper is not None:
            self.nonlinear_violation = _immutable_copy(
                np.maximum(
                    np.where(self.nonlinear_lower < 0.0, -self.nonlinear_lower, 0.0),
                    np.where(self.nonlinear_upper > 0.0, self.nonlinear_upper, 0.0),
                )
            )

    @classmethod
    def create(
        cls,
        context: EnOptContext,
        variables: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> ConstraintInfo | None:
        """Create a `ConstraintInfo` object with constraint difference data.

        This calculates differences between variables/constraints and their
        bounds. Differences for non-linear constraints are optional. All fields
        default to `None` and are only populated if bounds are present and finite.

        Args:
            context:      The optimizer context containing bound definitions.
            variables:    Variable values to check against bounds.
            constraints:  Non-linear constraint values, if present.

        Returns:
            A newly created `ConstraintInfo` object, or `None` if no bounds
            are available.
        """
        bound_lower: NDArray[np.float64] | None = None
        bound_upper: NDArray[np.float64] | None = None
        linear_lower: NDArray[np.float64] | None = None
        linear_upper: NDArray[np.float64] | None = None
        nonlinear_lower: NDArray[np.float64] | None = None
        nonlinear_upper: NDArray[np.float64] | None = None

        have_constraint_info = False

        if np.all(np.isfinite(context.variables.lower_bounds)) or np.all(
            np.isfinite(context.variables.upper_bounds)
        ):
            bound_lower = variables - context.variables.lower_bounds
            bound_upper = variables - context.variables.upper_bounds
            have_constraint_info = True

        if context.linear_constraints is not None:
            values = np.matmul(context.linear_constraints.coefficients, variables)
            linear_lower = values - context.linear_constraints.lower_bounds
            linear_upper = values - context.linear_constraints.upper_bounds
            have_constraint_info = True

        if constraints is not None:
            assert context.nonlinear_constraints is not None
            lower_bounds, upper_bounds = _get_nonlinear_constraint_bounds(context)
            nonlinear_lower = constraints - lower_bounds
            nonlinear_upper = constraints - upper_bounds
            have_constraint_info = True

        if have_constraint_info:
            return ConstraintInfo(
                bound_lower=bound_lower,
                bound_upper=bound_upper,
                linear_lower=linear_lower,
                linear_upper=linear_upper,
                nonlinear_lower=nonlinear_lower,
                nonlinear_upper=nonlinear_upper,
            )

        return None

    def _transform_from_optimizer(self, context: EnOptContext) -> ConstraintInfo:
        if (
            not context.variable_transforms
            and not context.nonlinear_constraint_transforms
        ):
            return self

        bound_lower: NDArray[np.float64] | None = self.bound_lower
        bound_upper: NDArray[np.float64] | None = self.bound_upper
        if bound_lower is not None:
            assert bound_upper is not None
            for variable_transform in context.variable_transforms:
                bound_lower, bound_upper = (
                    variable_transform.bound_constraint_diffs_from_optimizer(
                        bound_lower, bound_upper
                    )
                )
        linear_lower: NDArray[np.float64] | None = self.linear_lower
        linear_upper: NDArray[np.float64] | None = self.linear_upper
        if linear_lower is not None:
            assert linear_upper is not None
            for variable_transform in context.variable_transforms:
                linear_lower, linear_upper = (
                    variable_transform.linear_constraints_diffs_from_optimizer(
                        linear_lower, linear_upper
                    )
                )

        nonlinear_lower: NDArray[np.float64] | None = self.nonlinear_lower
        nonlinear_upper: NDArray[np.float64] | None = self.nonlinear_upper
        if nonlinear_lower is not None:
            assert nonlinear_upper is not None
            for constraint_transform in context.nonlinear_constraint_transforms:
                nonlinear_lower, nonlinear_upper = (
                    constraint_transform.nonlinear_constraint_diffs_from_optimizer(
                        nonlinear_lower, nonlinear_upper
                    )
                )

        return ConstraintInfo(
            bound_lower=bound_lower,
            bound_upper=bound_upper,
            linear_lower=linear_lower,
            linear_upper=linear_upper,
            nonlinear_lower=nonlinear_lower,
            nonlinear_upper=nonlinear_upper,
        )


def _get_nonlinear_constraint_bounds(
    context: EnOptContext,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    assert context.nonlinear_constraints is not None
    lower_bounds = context.nonlinear_constraints.lower_bounds
    upper_bounds = context.nonlinear_constraints.upper_bounds
    for constraint_transform in context.nonlinear_constraint_transforms:
        lower_bounds, upper_bounds = constraint_transform.bounds_to_optimizer(
            lower_bounds, upper_bounds
        )
    return lower_bounds, upper_bounds
