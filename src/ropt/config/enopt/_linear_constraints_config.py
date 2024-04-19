"""Configuration class for linear constraints."""

from __future__ import annotations

from pydantic import model_validator

from ropt.config.utils import (
    Array1D,
    Array2D,
    ArrayEnum,
    broadcast_1d_array,
    check_enum_values,
)
from ropt.enums import ConstraintType

from ._enopt_base_model import EnOptBaseModel


class LinearConstraintsConfig(EnOptBaseModel):
    r"""The configuration class for linear constraints.

    This class defines linear constraints configured by the `linear_constraints`
    field in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Linear constraints can be described by a set of linear equations on the
    variables, including equality or non-equality constraints. The
    `coefficients` field is a 2D `numpy` array where the number of rows equals
    the number of constraints, and the number of columns equals the number of
    variables.

    The right-hand sides of the equations are given in the `rhs_values` field,
    which will be converted and broadcasted to a `numpy` array with a length
    equal to the number of equations.

    The `types` field determines the type of each equation: equality ($=$) or
    inequality ($\leq$ or $\geq$), and it is broadcasted to a length equal to
    the number of equations. The `types` field is defined as an integer array,
    but its values are limited to those of the
    [`ConstraintType`][ropt.enums.ConstraintType] enumeration.

    Attributes:
        coefficients: The matrix of coefficients
        rhs_values:   The right-hand-sides of the constraint equations
        types:        The type of each equation
                      (see [`ConstraintType`][ropt.enums.ConstraintType])

    Info: Rescaled variables
        If a `LinearConstraintsConfig` object is part of an
        [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object, it may be
        modified during initialization. If the `EnOptConfig` object defines a
        rescaling of the variables, the linear coefficients ($\mathbf{A}$) and
        offsets ($\mathbf{b}$) are converted to remain valid for the scaled
        variables:

        $$
        \begin{align}
            \hat{\mathbf{A}} &= \mathbf{A}\mathbf{S} \\
            \hat{\mathbf{b}} &= \mathbf{b} - \mathbf{A}\mathbf{o}
        \end{align}
        $$

        where $\mathbf{S}$ is a diagonal matrix containing the variable scales,
        and $\mathbf{o}$ is a vector containing the variable offsets.

        It is important to realize that this does not mean that the constraints
        themselves are scaled. The equations and right-hand values are
        transformed, ensuring they yield the same results with scaled variables
        as the original equations and right-hand side values would with unscaled
        variables. This should be taken into account when comparing the
        difference between the equations and right-hand side to a tolerance.
    """

    coefficients: Array2D
    rhs_values: Array1D
    types: ArrayEnum

    @model_validator(mode="after")
    def _broadcast_and_check(self) -> LinearConstraintsConfig:
        size = 0 if self.coefficients is None else self.coefficients.shape[0]
        self.rhs_values = broadcast_1d_array(self.rhs_values, "rhs_values", size)
        check_enum_values(self.types, ConstraintType)
        self.types = broadcast_1d_array(self.types, "types", size)
        return self
