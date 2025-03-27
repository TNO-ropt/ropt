"""Configuration class for variables."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, ValidationInfo, model_validator

from ropt.config.utils import (
    ImmutableBaseModel,
    broadcast_1d_array,
    check_enum_values,
    immutable_array,
)
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DBool,
    ArrayEnum,
)
from ropt.enums import VariableType


class VariablesConfig(ImmutableBaseModel):
    r"""Configuration class for optimization variables.

    This class, `VariablesConfig`, defines the configuration for the
    optimization variables used in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. It specifies the
    initial values, bounds, types, and an optional mask for the variables.

    The `initial_values` field is a required `numpy` array that sets the
    starting values for the variables. The number of variables is determined by
    the length of this array.

    The `lower_bounds` and `upper_bounds` fields define the bounds for each
    variable. These are also `numpy` arrays and are broadcasted to match the
    number of variables. By default, they are set to negative and positive
    infinity, respectively. `numpy.nan` values in these arrays indicate
    unbounded variables and are converted to `numpy.inf` with the appropriate
    sign.

    The optional `types` field allows assigning a
    [`VariableType`][ropt.enums.VariableType] to each variable. If not provided,
    all variables are assumed to be continuous real-valued
    ([`VariableType.REAL`][ropt.enums.VariableType.REAL]).

    The optional `mask` field is a boolean `numpy` array that indicates which
    variables are free to change during optimization. `True` values in the mask
    indicate that the corresponding variable is free, while `False` indicates a
    fixed variable.

    Attributes:
        types:          Optional variable types.
        initial_values: Initial values for the variables.
        lower_bounds:   Lower bounds for the variables (default: $-\infty$).
        upper_bounds:   Upper bounds for the variables (default: $+\infty$).
        mask:           Optional boolean mask indicating free variables.
    """

    types: ArrayEnum | None = None
    initial_values: Array1D = np.array(0.0)
    lower_bounds: Array1D = np.array(-np.inf)
    upper_bounds: Array1D = np.array(np.inf)
    mask: Array1DBool | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_transform(self, info: ValidationInfo) -> Self:
        self._mutable()

        lower_bounds = broadcast_1d_array(
            self.lower_bounds, "lower_bounds", self.initial_values.size
        )
        upper_bounds = broadcast_1d_array(
            self.upper_bounds, "upper_bounds", self.initial_values.size
        )

        if info.context is not None and info.context.variables is not None:
            self.initial_values = immutable_array(
                info.context.variables.to_optimizer(self.initial_values)
            )
            lower_bounds = info.context.variables.to_optimizer(lower_bounds)
            upper_bounds = info.context.variables.to_optimizer(upper_bounds)

        if np.any(lower_bounds > upper_bounds):
            msg = "The lower bounds are larger than the upper bounds."
            raise ValueError(msg)

        self.lower_bounds = immutable_array(lower_bounds)
        self.upper_bounds = immutable_array(upper_bounds)

        if self.types is not None:
            check_enum_values(self.types, VariableType)
            self.types = broadcast_1d_array(
                self.types, "types", self.initial_values.size
            )
        if self.mask is not None:
            self.mask = broadcast_1d_array(self.mask, "mask", self.initial_values.size)

        self._immutable()

        return self
