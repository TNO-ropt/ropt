"""Configuration class for linear constraints."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from ropt.config.utils import broadcast_1d_array, immutable_array
from ropt.config.validated_types import Array1D, Array2D  # noqa: TC001


class LinearConstraintsConfig(BaseModel):
    r"""Configuration class for linear constraints.

    This class, `LinearConstraintsConfig`, defines linear constraints used in an
    optimization, for instance as part of an
    [`EnOptConfig`][ropt.config.EnOptConfig] object.

    Linear constraints are defined by a set of linear equations involving the
    optimization variables. These equations can represent equality or inequality
    constraints. The `coefficients` field is a 2D `numpy` array where each row
    represents a constraint, and each column corresponds to a variable.

    The `lower_bounds` and `upper_bounds` fields specify the bounds on the
    right-hand side of each constraint equation. These fields are converted and
    broadcasted to `numpy` arrays with a length equal to the number of
    constraint equations.

    Less-than and greater-than inequality constraints can be specified by
    setting the lower bounds to $-\infty$, or the upper bounds to $+\infty$,
    respectively. Equality constraints are specified by setting the lower bounds
    equal to the upper bounds.

    Attributes:
        coefficients: Matrix of coefficients for the linear constraints.
        lower_bounds: Lower bounds for the right-hand side of the constraint equations.
        upper_bounds: Upper bounds for the right-hand side of the constraint equations.

    Note: Linear transformation of variables.
        The set of linear constraints can be represented by a matrix equation:
        $\mathbf{A} \mathbf{x} = \mathbf{b}$.

        When linearly transforming variables to the optimizer domain, the
        coefficients ($\mathbf{A}$) and right-hand-side values ($\mathbf{b}$)
        must be converted to remain valid. If the linear transformation of the
        variables to the optimizer domain is given by:

        $$ \hat{\mathbf{x}} = \mathbf{S} \mathbf{x} + \mathbf{o}$$

        then the coefficients and right-hand-side values must be transformed as
        follows:

        $$ \begin{align}
            \hat{\mathbf{A}} &= \mathbf{A} \mathbf{S}^{-1} \\ \hat{\mathbf{b}}
            &= \mathbf{b} + \mathbf{A}\mathbf{S}^{-1}\mathbf{o}
        \end{align}$$
    """

    coefficients: Array2D
    lower_bounds: Array1D
    upper_bounds: Array1D

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(self, info: ValidationInfo) -> Self:
        coefficients = self.coefficients
        size = 0 if coefficients is None else coefficients.shape[0]
        lower_bounds = broadcast_1d_array(self.lower_bounds, "lower_bounds", size)
        upper_bounds = broadcast_1d_array(self.upper_bounds, "upper_bounds", size)

        if info.context is not None and info.context.variables is not None:
            coefficients, lower_bounds, upper_bounds = (
                info.context.variables.linear_constraints_to_optimizer(
                    coefficients, lower_bounds, upper_bounds
                )
            )
            coefficients = immutable_array(coefficients)
            lower_bounds = immutable_array(lower_bounds)
            upper_bounds = immutable_array(upper_bounds)

        if np.any(lower_bounds > upper_bounds):
            msg = "The lower bounds are larger than the upper bounds."
            raise ValueError(msg)

        return self.model_copy(
            update={
                "coefficients": coefficients,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
            }
        )
