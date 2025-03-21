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
    r"""Configuration class for non-linear constraints.

    This class, `NonlinearConstraintsConfig`, defines non-linear constraints used
    in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Non-linear constraints are defined by comparing a constraint function to a
    right-hand-side value, allowing for equality or inequality constraints. The
    `lower_bounds` and `upper_bounds` fields, which are `numpy` arrays, specify the
    bounds on these right-hand-side values. The length of these arrays determines
    the number of constraint functions.

    Less-than and greater-than inequality constraints can be specified by
    setting the lower bounds to $-\infty$, or the upper bounds to $+\infty$,
    respectively. Equality constraints are specified by setting the lower bounds
    equal to the upper bounds.

    Non-linear constraints can be processed by realization filters and function
    estimators. The `realization_filters` and `function_estimators` fields contain
    indices that refer to the corresponding objects configured in the parent
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Attributes:
        lower_bounds:        Lower bounds for the right-hand-side values.
        upper_bounds:        Upper bounds for the right-hand-side values.
        realization_filters: Optional indices of realization filters.
        function_estimators: Optional indices of function estimators.
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

        if np.any(lower_bounds > upper_bounds):
            msg = "The non-linear constraint lower bounds are larger than the upper bounds."
            raise ValueError(msg)

        self._mutable()
        self.lower_bounds = immutable_array(lower_bounds)
        self.upper_bounds = immutable_array(upper_bounds)
        self._immutable()

        return self
