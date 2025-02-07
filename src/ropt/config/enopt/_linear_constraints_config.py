"""Configuration class for linear constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import ConfigDict, model_validator

from ropt.config.utils import ImmutableBaseModel, broadcast_1d_array, check_enum_values
from ropt.config.validated_types import Array1D, Array2D, ArrayEnum  # noqa: TC001
from ropt.enums import ConstraintType

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptContext, VariablesConfig


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
        self, variables: VariablesConfig, context: EnOptContext | None
    ) -> LinearConstraintsConfig:
        """Transform linear constraints.

        Args:
            variables: A variables configuration object specifying.
            context:   The configuration context.

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

        if context is not None and context.transforms.variables is not None:
            coefficients, rhs_values = (
                context.transforms.variables.transform_linear_constraints(
                    self.coefficients, self.rhs_values
                )
            )
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
