"""Configuration class for linear constraints."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ConfigDict, model_validator

from ropt.config.utils import ImmutableBaseModel, broadcast_1d_array, check_enum_values
from ropt.config.validated_types import Array1D, Array2D, ArrayEnum  # noqa: TCH001
from ropt.enums import ConstraintType

if TYPE_CHECKING:
    from ropt.config.enopt import VariablesConfig

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class LinearConstraintsConfig(ImmutableBaseModel):
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
        coefficients: The matrix of coefficients.
        rhs_values:   The right-hand-sides of the constraint equations.
        types:        The type of each equation
                      (see [`ConstraintType`][ropt.enums.ConstraintType]).

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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(self) -> Self:
        self._mutable()
        size = 0 if self.coefficients is None else self.coefficients.shape[0]
        self.rhs_values = broadcast_1d_array(self.rhs_values, "rhs_values", size)
        check_enum_values(self.types, ConstraintType)
        self.types = broadcast_1d_array(self.types, "types", size)
        self._immutable()
        return self

    def apply_transformation(
        self, variables: VariablesConfig
    ) -> LinearConstraintsConfig:
        """Transform linear constraints with variable offsets and scales.

        If offsets and/or scales are specified in the variables configuration,
        the linear constraints need adjustment to maintain their validity after
        variable transformation. This method returns a new linear constraints
        configuration with these adjustments applied.

        Args:
            variables: A variables configuration object specifying.

        Returns:
            A modified configuration if transformations are applied; otherwise, self.
        """
        variable_count = variables.initial_values.size
        if (
            self.coefficients.shape[0] > 0
            and self.coefficients.shape[1] != variable_count
        ):
            msg = f"the coefficients matrix should have {variable_count} columns"
            raise ValueError(msg)

        # Correct the linear system of input constraints for scaling:
        if variables.offsets is not None or variables.scales is not None:
            coefficients = self.coefficients
            rhs_values = self.rhs_values
            if variables.offsets is not None:
                rhs_values = rhs_values - np.matmul(coefficients, variables.offsets)
            if variables.scales is not None:
                coefficients = coefficients * variables.scales
            values = self.model_dump(round_trip=True)
            values.update(
                coefficients=coefficients,
                rhs_values=rhs_values,
                types=self.types,
            )
            return LinearConstraintsConfig.model_construct(
                **values,
            )
        return self
