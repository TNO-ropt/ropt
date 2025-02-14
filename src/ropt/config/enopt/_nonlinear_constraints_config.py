"""Configuration class for non-linear constraints."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, ValidationInfo, model_validator

from ropt.config.utils import ImmutableBaseModel, broadcast_arrays, immutable_array
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DInt,
)


class NonlinearConstraintsConfig(ImmutableBaseModel):
    r"""The configuration class for non-linear constraints.

    This class defines non-linear constraints configured by the
    `nonlinear_constraints` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Non-linear constraints require that some constraint function is compared to
    a right-hand-side value, either for equality or inequality. The
    `lower_bounds` and `upper_bounds` fields, which is a `numpy` arrays with a
    length equal to the number of constraint functions, provides the bounds on
    the right-hand-side values.

    The non-linear constraints may be subject to realization filters and
    function estimators. The `realization_filters` and `function_estimator`
    fields contain indices to the realization filter or function estimator
    objects to use. These objects are configured in the parent
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Attributes:
        lower_bounds:        The lower bounds on the right-hand-side values.
        upper_bounds:        The upper bounds on the right-hand-side values.
        realization_filters: Optional realization filter indices.
        function_estimators: Optional function estimator indices.
    """

    lower_bounds: Array1D
    upper_bounds: Array1D
    realization_filters: Array1DInt | None = None
    function_estimators: Array1DInt | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(self, info: ValidationInfo) -> Self:
        lower_bounds, upper_bounds = broadcast_arrays(
            self.lower_bounds, self.upper_bounds
        )

        if info.context is not None and info.context.nonlinear_constraints is not None:
            lower_bounds, upper_bounds = (
                info.context.nonlinear_constraints.bounds_to_optimizer(
                    lower_bounds, upper_bounds
                )
            )

        self._mutable()
        self.lower_bounds = immutable_array(
            np.where(lower_bounds < upper_bounds, lower_bounds, upper_bounds)
        )
        self.upper_bounds = immutable_array(
            np.where(upper_bounds > lower_bounds, upper_bounds, lower_bounds)
        )
        self._immutable()

        return self
