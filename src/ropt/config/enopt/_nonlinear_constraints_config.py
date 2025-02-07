"""Configuration class for non-linear constraints."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, ValidationInfo, model_validator

from ropt.config.utils import ImmutableBaseModel, broadcast_1d_array, check_enum_values
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
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

    Attributes:
        rhs_values:          The right-hand-side values.
        types:               The type of each non-linear constraint.
                             ([`ConstraintType`][ropt.enums.ConstraintType]).
        realization_filters: Optional realization filter indices.
        function_estimators: Optional function estimator indices.
    """

    rhs_values: Array1D
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
