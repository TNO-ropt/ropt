"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, model_validator

from ropt.config.utils import ImmutableBaseModel, normalize
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DInt,
)


class ObjectiveFunctionsConfig(ImmutableBaseModel):
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

    Objective functions can be processed by realization filters and function
    estimators. The `realization_filters` and `function_estimators` fields
    contain indices that refer to the corresponding configuration objects in the
    parent [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

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
        self._mutable()
        self.weights = normalize(self.weights)
        self._immutable()
        return self
