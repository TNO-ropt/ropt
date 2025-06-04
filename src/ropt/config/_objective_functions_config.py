"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt.config.utils import immutable_array, normalize
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DInt,
)


class ObjectiveFunctionsConfig(BaseModel):
    """Configuration class for objective functions.

    This class, `ObjectiveFunctionsConfig`, defines the configuration for
    objective functions. for instance, as part of an
    [`EnOptConfig`][ropt.config.EnOptConfig] object.

    `ropt` supports multi-objective optimization. Multiple objectives are
    combined into a single value by summing them after weighting. The `weights`
    field, a `numpy` array, determines the weight of each objective function.
    The length of this array defines the number of objective functions. The
    weights are automatically normalized to sum to 1 (e.g., `[1, 1]` becomes
    `[0.5, 0.5]`).

    Objective functions can optionally be processed using [`realization
    filters`][ropt.config.RealizationFilterConfig] and [`function
    estimators`][ropt.config.FunctionEstimatorConfig].The `realization_filters`
    and `function_estimators` attributes, if provided, must be arrays of integer
    indices. Each index in the `realization_filters` array corresponds to a
    objective (by position) and specifies which filter to use. The available
    filters must be defined elsewhere as a tuple of realization filter
    configurations. For instance, for optimization these are defined in the
    [`EnOptConfig.realization_filters`][ropt.config.EnOptConfig] configuration
    class. The same logic applies to the `function_estimators` array . If an
    index is invalid (e.g., out of bounds for the corresponding object tuple),
    no filter or estimator is applied to that specific objective. If these
    attributes are not provided (`None`), no filters or estimators are applied
    at all.

    Attributes:
        weights:             Weights for the objective functions (default: 1.0).
        realization_filters: Optional indices of realization filters.
        function_estimators: Optional indices of function estimators.
    """

    weights: Array1D = np.array(1.0)
    realization_filters: Array1DInt = np.array(-1)
    function_estimators: Array1DInt = np.array(0)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_normalize(self) -> Self:
        self.weights = normalize(self.weights)
        self.realization_filters = immutable_array(
            np.broadcast_to(self.realization_filters, self.weights.shape)
        )
        self.function_estimators = immutable_array(
            np.broadcast_to(self.function_estimators, self.weights.shape)
        )
        return self
