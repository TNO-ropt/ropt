"""Configuration class for non-linear constraints."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, model_validator

from ropt.config.utils import (
    ImmutableBaseModel,
    broadcast_1d_array,
    broadcast_arrays,
    check_enum_values,
)
from ropt.config.validated_types import (  # noqa: TCH001
    Array1D,
    Array1DBool,
    Array1DInt,
    ArrayEnum,
    UniqueNames,
)
from ropt.enums import ConstraintType


class NonlinearConstraintsConfig(ImmutableBaseModel):
    r"""The configuration class for non-linear constraints.

    This class defines non-linear constraints configured by the
    `nonlinear_constraints` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Non-linear constraints require that some constraint function is compared to
    a right-hand-side value, either for equality or inequality. The `rhs_values`
    field, which is a `numpy` array with a length equal to the number of
    constraint functions, provides the right-hand-side values.

    The `names` field is optional. If given, the number of constraint functions
    is set equal to its length. The `rhs_values` array will then be broadcasted
    to the number of constraint functions. For example, if `names = ["c1",
    "c2"]` and `rhs_values = 0.0`, the optimizer assumes two constraint
    functions and stores `rhs_values = [0.0, 0.0]`. If `names` is not set, the
    number of constraints is determined by the length of `rhs_values`.

    The `scales` field contains scaling values for the constraints. These values
    scale the constraint function values to a desired order of magnitude. Each
    time new constraint function values are obtained during optimization, they
    are divided by these values. The `auto_scale` field can be used to direct
    the optimizer to obtain additional scaling by multiplying the values of
    `scales` by the values of the constraint functions at the start of the
    optimization. Both the `scales` and `auto_scale` arrays will be broadcasted
    to have a size equal to the number of constraint functions.

    Info: Manual scaling and auto_scaling
        Both the `scales` values and the values obtained by auto-scaling will be
        applied. If `scales` is not supplied, auto-scaling will scale the
        constraints such that their initial values will be equal to one. Setting
        `scales` additionally allows for scaling to different initial values.

    The `types` field determines the type of each constraint: equality ($=$) or
    inequality ($\le$ or $\ge$), and is broadcasted to a length equal to the
    number of constraints. The `types` field is defined as an integer array, but
    its values are limited to those of the
    [`ConstraintType`][ropt.enums.ConstraintType] enumeration.

    The non-linear constraints may be subject to realization filters and
    function transforms. The `realization_filters` and `function_transform`
    fields contain indices to the realization filter or function transform
    objects to use. These objects are configured in the parent
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Info: Calculating auto-scaling values
        For auto-scaling, the initial value of the constraint functions is used
        based on the assumption that it is calculated as a weighted sum from an
        ensemble of realizations, with weights specified by the `realizations`
        field in the [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. If
        the `realizations_filters` and/or `function_transforms` fields are also
        set, this assumption may not strictly hold; however, the scaling value
        will still be calculated in the same way for practical purposes.

    Attributes:
        names:               The names of the constraint functions (optional).
        rhs_values:          The right-hand-side values.
        scales:              The scaling factors (default: 1.0).
        auto_scale:          Enable/disable auto-scaling (default: `False`).
        types:               The type of each non-linear constraint.
                             ([`ConstraintType`][ropt.enums.ConstraintType]).
        realization_filters: Optional realization filter indices.
        function_transforms: Optional function transform indices.
    """

    names: UniqueNames | None = None
    rhs_values: Array1D
    scales: Array1D = np.array(1.0)
    auto_scale: Array1DBool = np.array([False])
    types: ArrayEnum
    realization_filters: Array1DInt | None = None
    function_transforms: Array1DInt | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(
        self,
    ) -> Self:
        self._mutable()
        if self.names is not None:
            size = len(self.names)
            for name in ("rhs_values", "scales", "auto_scale"):
                setattr(self, name, broadcast_1d_array(getattr(self, name), name, size))
        else:
            self.rhs_values, self.scales, self.auto_scale = broadcast_arrays(
                self.rhs_values, self.scales, self.auto_scale
            )

        check_enum_values(self.types, ConstraintType)
        self.types = broadcast_1d_array(self.types, "types", self.rhs_values.size)
        self._immutable()
        return self
