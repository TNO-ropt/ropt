"""Configuration class for linear constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from pydantic import ConfigDict, model_validator

from ropt.config.utils import (
    ImmutableBaseModel,
    broadcast_1d_array,
    immutable_array,
)
from ropt.config.validated_types import Array1D, Array2D  # noqa: TC001

if TYPE_CHECKING:
    from ropt.config.enopt import VariablesConfig
    from ropt.transforms import OptModelTransforms


class LinearConstraintsConfig(ImmutableBaseModel):
    r"""The configuration class for linear constraints.

    This class defines linear constraints configured by the `linear_constraints`
    field in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Linear constraints can be described by a set of linear equations on the
    variables, including equality or non-equality constraints. The
    `coefficients` field is a 2D `numpy` array where the number of rows equals
    the number of constraints, and the number of columns equals the number of
    variables.

    Lower and upper bounds on he right-hand sides of the equations are given in
    the `lower_bounds` and `upper_bounds fields, which will be converted and
    broadcasted to a `numpy` array with a length equal to the number of
    equations.

    Attributes:
        coefficients: The matrix of coefficients.
        lower_bounds: The lower bounds on the right-hand-sides of the constraint equations.
        upper_bounds: The upper bounds on the right-hand-sides of the constraint equations.
    """

    coefficients: Array2D
    lower_bounds: Array1D
    upper_bounds: Array1D

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(self) -> Self:
        size = 0 if self.coefficients is None else self.coefficients.shape[0]
        lower_bounds = broadcast_1d_array(self.lower_bounds, "lower_bounds", size)
        upper_bounds = broadcast_1d_array(self.upper_bounds, "upper_bounds", size)
        self._mutable()
        self.lower_bounds = immutable_array(
            np.where(lower_bounds < upper_bounds, lower_bounds, upper_bounds)
        )
        self.upper_bounds = immutable_array(
            np.where(upper_bounds > lower_bounds, upper_bounds, lower_bounds)
        )
        self._immutable()
        return self

    def apply_transformation(
        self, variables: VariablesConfig, transforms: OptModelTransforms | None
    ) -> LinearConstraintsConfig:
        """Transform linear constraints.

        Args:
            variables:  A variables configuration object specifying.
            transforms: An optional transforms object.

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

        if transforms is not None and transforms.variables is not None:
            coefficients, lower_bounds, upper_bounds = (
                transforms.variables.transform_linear_constraints(
                    self.coefficients, self.lower_bounds, self.upper_bounds
                )
            )
            values = self.model_dump(round_trip=True)
            values.update(
                coefficients=coefficients,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            return LinearConstraintsConfig.model_construct(**values)

        return self
