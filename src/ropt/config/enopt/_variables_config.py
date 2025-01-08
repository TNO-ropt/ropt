"""Configuration class for variables."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from ropt.config.utils import (
    ImmutableBaseModel,
    broadcast_1d_array,
    check_enum_values,
    immutable_array,
)
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    ArrayEnum,
    ArrayIndices,
    UniqueNames,
)
from ropt.enums import VariableType
from ropt.utils.misc import format_tuple


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

    The `names` field is optional since variable names are not strictly needed
    by the optimizer. If given, the lenght of this field must be equal to the number
    of variables.

    Info: Variable Names
        `ropt` does not use the names itself, and names can be of arbitrary
        type, as long as they are unique. However, some optimizers or external
        code might need a string representation of each name, which can be
        obtained using the
        [`get_formatted_names`][ropt.config.enopt.VariablesConfig.get_formatted_names]
        method. The `delimiters` attribute is used by this method to convert the
        special case of names consisting of tuples of strings.

    The optional `types` field can be used to assign types to each variable,
    according to the [`VariableType`][ropt.enums.VariableType] enumeration. The
    values can be used to configure the optimizer accordingly. If not provided,
    all variables are assumed to be continuous and of real data type
    (corresponding to [`VariableType.REAL`][ropt.enums.VariableType.REAL])

    The `offsets` and `scales` fields are optional: if given, they are
    broadcasted to the number of variables and used for scaling. The elements
    $x_i$ of `initial_values`, `lower_bounds`, and `upper_bounds` fields are
    rescaled by the elements $o_i$ and $s_i$ of `offsets` and `scales`: $(x_i -
    o_i) / s_i$.

    Info: Transformation of Linear Constraints
        Any linear constraints defined in the `EnOptConfig` object via its
        `linear_constraints` field will also be transformed using any offsets
        and scales passed via the `VariablesConfig` object. See:
        [`LinearConstraintsConfig`][ropt.config.enopt.LinearConstraintsConfig].

    The optional `indices` field contains the indices of the variables
    considered to be free to change. During optimization, only these variables
    should change while others remain fixed.

    Attributes:
        names:          Optional names of the variables.
        types:          The type of the variables (optional).
        initial_values: The initial values of the variables.
        lower_bounds:   Lower bound of the variables (default: $-\infty$).
        upper_bounds:   Upper bound of the variables (default: $+\infty$).
        offsets:        Optional offsets, used for scaling the variables.
        scales:         Optional scales, used for scaling the variables.
        indices:        Optional indices of variables to optimize.
        delimiters:     Delimiters used to construct names from tuples.
    """

    names: UniqueNames | None = None
    types: ArrayEnum | None = None
    initial_values: Array1D = np.array(0.0)
    lower_bounds: Array1D = np.array(-np.inf)
    upper_bounds: Array1D = np.array(np.inf)
    offsets: Array1D | None = None
    scales: Array1D | None = None
    indices: ArrayIndices | None = None
    delimiters: str = Field(":", min_length=0)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_scale(self) -> Self:
        self._mutable()

        self.lower_bounds = broadcast_1d_array(
            self.lower_bounds, "lower_bounds", self.initial_values.size
        )
        self.upper_bounds = broadcast_1d_array(
            self.upper_bounds, "upper_bounds", self.initial_values.size
        )

        if self.offsets is not None:
            self.offsets = broadcast_1d_array(
                self.offsets, "offsets", self.initial_values.size
            )
            self.initial_values = immutable_array(self.initial_values - self.offsets)
            self.lower_bounds = immutable_array(self.lower_bounds - self.offsets)
            self.upper_bounds = immutable_array(self.upper_bounds - self.offsets)
        if self.scales is not None:
            self.scales = broadcast_1d_array(
                self.scales, "scales", self.initial_values.size
            )
            self.initial_values = immutable_array(self.initial_values / self.scales)
            self.lower_bounds = immutable_array(self.lower_bounds / self.scales)
            self.upper_bounds = immutable_array(self.upper_bounds / self.scales)

        if self.types is not None:
            check_enum_values(self.types, VariableType)
            self.types = broadcast_1d_array(
                self.types, "types", self.initial_values.size
            )

        self._immutable()

        return self

    def get_formatted_names(self) -> tuple[str, ...] | None:
        """Return string representations of the variable names.

        This method converts the variable names to a tuple of strings. Each name
        is converted using its string representation unless the name is a tuple.
        In that case, the tuple items are converted to strings and joined using
        the delimiters taken from the `delimiter` field in the `Variables`
        object. This field is a string that may consist of multiple delimiters
        used in turn. If it contains fewer items than needed, the last one is
        used for the missing ones. By default, the `:` character is used as the
        delimiter.

        Returns:
            A tuple of formatted variable names.
        """
        if self.names is None:
            return None

        return tuple(format_tuple(name, self.delimiters) for name in self.names)
