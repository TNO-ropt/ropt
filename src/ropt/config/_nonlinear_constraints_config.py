"""Configuration class for non-linear constraints."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt._utils import broadcast_1d_array, broadcast_arrays

from ._validated_types import (  # noqa: TC001
    Array1D,
    Array1DInt,
)


class NonlinearConstraintsConfig(BaseModel):
    r"""Configuration class for non-linear constraints.

    `NonlinearConstraintsConfig` defines nonlinear constraints used as the
    `nonlinear_constraints` field of an
    [`EnOptContext`][ropt.context.EnOptContext] object.

    Nonlinear constraints are defined by comparing a constraint function to a
    right-hand-side value, allowing for equality or inequality constraints. The
    `lower_bounds` and `upper_bounds` fields, which are `numpy` arrays, specify
    the bounds on these right-hand-side values. The length of these arrays
    determines the number of constraint functions.

    Less-than and greater-than inequality constraints can be specified by
    setting the lower bounds to $-\infty$, or the upper bounds to $+\infty$,
    respectively. Equality constraints are specified by setting the lower bounds
    equal to the upper bounds.

    Constraint functions can optionally be processed using [`realization
    filters`][ropt.realization_filter.RealizationFilter] and [`function
    estimators`][ropt.function_estimator.FunctionEstimator] objects. The
    `realization_filters` and `function_estimators` fields are integer index
    arrays: each entry selects a filter or estimator by its position in the
    corresponding tuple defined in [`EnOptContext`][ropt.context.EnOptContext].
    An out-of-range index means no filter or estimator is applied to that
    constraint. If a field uses its default value of `-1`, no filter or
    estimator is applied at all.

    Attributes:
        lower_bounds:        Lower bounds for the right-hand-side values.
        upper_bounds:        Upper bounds for the right-hand-side values.
        realization_filters: Optional indices of realization filters.
        function_estimators: Optional indices of function estimators.
        transforms:          Optional indices of constraint transforms.
    """

    lower_bounds: Array1D
    upper_bounds: Array1D
    realization_filters: Array1DInt = np.array(-1)
    function_estimators: Array1DInt = np.array(0)
    transforms: Array1DInt = np.array(-1)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(self) -> Self:
        lower_bounds, upper_bounds = broadcast_arrays(
            self.lower_bounds, self.upper_bounds
        )
        return self.model_copy(
            update={
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "realization_filters": broadcast_1d_array(
                    self.realization_filters, "realization_filters", lower_bounds.size
                ),
                "function_estimators": broadcast_1d_array(
                    self.function_estimators, "function_estimators", lower_bounds.size
                ),
                "transforms": broadcast_1d_array(
                    self.transforms, "transforms", lower_bounds.size
                ),
            }
        )
