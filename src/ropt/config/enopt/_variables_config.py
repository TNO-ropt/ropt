"""Configuration class for variables."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, ValidationInfo, model_validator

from ropt.config.utils import (
    ImmutableBaseModel,
    broadcast_1d_array,
    check_enum_values,
)
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    ArrayEnum,
    ArrayIndices,
)
from ropt.enums import VariableType


class VariablesConfig(ImmutableBaseModel):
    r"""The configuration class for variables.

    This configuration class, configured by the `variables` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object defines essential
    aspects of the variables: the initial values and the bounds. These are given
    by the `initial_values`, `lower_bounds`, and `upper_bounds` fields, which
    are [`numpy`](https://numpy.org) arrays. Initial values must be provided,
    and its length determines the number of variables. The lower and upper
    bounds, are broadcasted to the number of variables, and are set to
    $-\infty$ and $+\infty$ by default. They may contain `numpy.nan` values,
    indicating that corresponding variables have no lower or upper bounds,
    respectively. These values are converted to `numpy.inf` values with an
    appropriate sign.

    The optional `types` field can be used to assign types to each variable,
    according to the [`VariableType`][ropt.enums.VariableType] enumeration. The
    values can be used to configure the optimizer accordingly. If not provided,
    all variables are assumed to be continuous and of real data type
    (corresponding to [`VariableType.REAL`][ropt.enums.VariableType.REAL])

    The optional `indices` field contains the indices of the variables
    considered to be free to change. During optimization, only these variables
    should change while others remain fixed.

    Attributes:
        types:          The type of the variables (optional).
        initial_values: The initial values of the variables.
        lower_bounds:   Lower bound of the variables (default: $-\infty$).
        upper_bounds:   Upper bound of the variables (default: $+\infty$).
        indices:        Optional indices of variables to optimize.
    """

    types: ArrayEnum | None = None
    initial_values: Array1D = np.array(0.0)
    lower_bounds: Array1D = np.array(-np.inf)
    upper_bounds: Array1D = np.array(np.inf)
    indices: ArrayIndices | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_transform(self, info: ValidationInfo) -> Self:
        self._mutable()

        self.lower_bounds = broadcast_1d_array(
            self.lower_bounds, "lower_bounds", self.initial_values.size
        )
        self.upper_bounds = broadcast_1d_array(
            self.upper_bounds, "upper_bounds", self.initial_values.size
        )

        if info.context is not None and info.context.transforms.variables is not None:
            self.initial_values = info.context.transforms.variables.forward(
                self.initial_values
            )
            self.lower_bounds = info.context.transforms.variables.forward(
                self.lower_bounds
            )
            self.upper_bounds = info.context.transforms.variables.forward(
                self.upper_bounds
            )

        if self.types is not None:
            check_enum_values(self.types, VariableType)
            self.types = broadcast_1d_array(
                self.types, "types", self.initial_values.size
            )

        self._immutable()

        return self
