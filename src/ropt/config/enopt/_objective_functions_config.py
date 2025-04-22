"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt.config.utils import normalize
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DInt,
)


class ObjectiveFunctionsConfig(BaseModel):
    """Configuration class for objective functions.

    This class, `ObjectiveFunctionsConfig`, defines the configuration for
    objective functions used in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    `ropt` supports multi-objective optimization. Multiple objectives are
    combined into a single value by summing them after weighting. The `weights`
    field, a `numpy` array, determines the weight of each objective function.
    The length of this array defines the number of objective functions. The
    weights are automatically normalized to sum to 1 (e.g., `[1, 1]` becomes
    `[0.5, 0.5]`).

    Objective functions can optionally be processed using realization filters
    and function estimators defined in the main
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig]. The `realization_filters`
    and `function_estimators` attributes, if provided, must be arrays of integer
    indices. Each index in the `realization_filters` array corresponds to an
    objective function (by position) and specifies which filter from the parent
    [`EnOptConfig.realization_filters`][ropt.config.enopt.EnOptConfig] tuple
    should be applied to it. The same logic applies to the `function_estimators`
    array and the parent
    [`EnOptConfig.function_estimators`][ropt.config.enopt.EnOptConfig] tuple. If
    an index is invalid (e.g., out of bounds for the corresponding parent
    tuple), no filter or estimator is applied to that specific objective
    function. If these attributes are not provided (`None`), no filters or
    estimators are applied at all.

    Attributes:
        weights:             Weights for the objective functions (default: 1.0).
        realization_filters: Optional indices of realization filters.
        function_estimators: Optional indices of function estimators.
    """

    weights: Array1D = np.array(1.0)
    realization_filters: Array1DInt | None = None
    function_estimators: Array1DInt | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_normalize(self) -> Self:
        self.weights = normalize(self.weights)
        return self
