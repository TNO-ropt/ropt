"""Configuration class for non-linear constraints."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from ropt.config.utils import broadcast_arrays, immutable_array
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DInt,
)


class NonlinearConstraintsConfig(BaseModel):
    r"""Configuration class for non-linear constraints.

    This class, `NonlinearConstraintsConfig`, defines non-linear constraints used
    in an [`EnOptConfig`][ropt.config.EnOptConfig] object.

    Non-linear constraints are defined by comparing a constraint function to a
    right-hand-side value, allowing for equality or inequality constraints. The
    `lower_bounds` and `upper_bounds` fields, which are `numpy` arrays, specify the
    bounds on these right-hand-side values. The length of these arrays determines
    the number of constraint functions.

    Less-than and greater-than inequality constraints can be specified by
    setting the lower bounds to $-\infty$, or the upper bounds to $+\infty$,
    respectively. Equality constraints are specified by setting the lower bounds
    equal to the upper bounds.

    Non-linear constraints can optionally be processed using realization filters
    and function estimators defined in the main
    [`EnOptConfig`][ropt.config.EnOptConfig]. The `realization_filters` and
    `function_estimators` attributes, if provided, must be arrays of integer
    indices. Each index in the `realization_filters` array corresponds to a
    constraint function (by position) and specifies which filter from the parent
    [`EnOptConfig.realization_filters`][ropt.config.EnOptConfig] tuple should be
    applied to it. The same logic applies to the `function_estimators` array and
    the parent [`EnOptConfig.function_estimators`][ropt.config.EnOptConfig]
    tuple. If an index is invalid (e.g., out of bounds for the corresponding
    parent tuple), no filter or estimator is applied to that specific constraint
    function. If these attributes are not provided (`None`), no filters or
    estimators are applied at all.

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
    def _broadcast_and_check(self) -> Self:
        lower_bounds, upper_bounds = broadcast_arrays(
            self.lower_bounds, self.upper_bounds
        )
        self.lower_bounds = immutable_array(lower_bounds)
        self.upper_bounds = immutable_array(upper_bounds)
        return self
