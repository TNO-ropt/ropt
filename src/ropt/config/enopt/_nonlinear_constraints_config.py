"""Configuration class for non-linear constraints."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, ValidationInfo, model_validator

from ropt.config.utils import ImmutableBaseModel, broadcast_1d_array, check_enum_values
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DBool,
    Array1DInt,
    ArrayEnum,
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
    function estimators. The `realization_filters` and `function_estimator`
    fields contain indices to the realization filter or function estimator
    objects to use. These objects are configured in the parent
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Info: Calculating auto-scaling values
        For auto-scaling, the initial value of the constraint functions is used
        based on the assumption that it is calculated as a weighted sum from an
        ensemble of realizations, with weights specified by the `realizations`
        field in the [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. If
        the `realizations_filters` and/or `function_estimators` fields are also
        set, this assumption may not strictly hold; however, the scaling value
        will still be calculated in the same way for practical purposes.

    Attributes:
        rhs_values:          The right-hand-side values.
        scales:              The scaling factors (default: 1.0).
        auto_scale:          Enable/disable auto-scaling (default: `False`).
        types:               The type of each non-linear constraint.
                             ([`ConstraintType`][ropt.enums.ConstraintType]).
        realization_filters: Optional realization filter indices.
        function_estimators: Optional function estimator indices.
    """

    rhs_values: Array1D
    scales: Array1D = np.array(1.0)
    auto_scale: Array1DBool = np.array([False])
    types: ArrayEnum
    realization_filters: Array1DInt | None = None
    function_estimators: Array1DInt | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(self, info: ValidationInfo) -> Self:
        self._mutable()

        self.scales = broadcast_1d_array(self.scales, "scales", self.rhs_values.size)
        self.auto_scale = broadcast_1d_array(
            self.auto_scale, "auto_scale", self.rhs_values.size
        )
        check_enum_values(self.types, ConstraintType)
        self.types = broadcast_1d_array(self.types, "types", self.rhs_values.size)

        if (
            info.context is not None
            and info.context.transforms.nonlinear_constraints is not None
        ):
            self.rhs_values, self.types = (
                info.context.transforms.nonlinear_constraints.transform_rhs_values(
                    self.rhs_values, self.types
                )
            )

        self._immutable()
        return self
