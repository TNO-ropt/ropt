"""Configuration class for linear constraints."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt._utils import broadcast_1d_array
from ropt.config._validated_types import Array1D, Array2D  # noqa: TC001


class LinearConstraintsConfig(BaseModel):
    r"""Configuration class for linear constraints.

    `LinearConstraintsConfig` defines linear constraints used as the
    `linear_constraints` field of an
    [`EnOptContext`][ropt.context.EnOptContext] object.

    See the [Configuration
    guide](../usage/configuration.md#linear_constraints) for detailed
    descriptions and usage examples.

    Attributes:
        coefficients: Matrix of coefficients for the linear constraints.
        lower_bounds: Lower bounds for the right-hand side of the constraint equations.
        upper_bounds: Upper bounds for the right-hand side of the constraint equations.
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
    def _broadcast_and_check(self) -> Self:
        coefficients = self.coefficients
        size = 0 if coefficients is None else coefficients.shape[0]
        lower_bounds = broadcast_1d_array(self.lower_bounds, "lower_bounds", size)
        upper_bounds = broadcast_1d_array(self.upper_bounds, "upper_bounds", size)

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
