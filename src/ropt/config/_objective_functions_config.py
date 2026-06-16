"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt._utils import broadcast_1d_array, normalize

from ._validated_types import (  # noqa: TC001
    Array1D,
    Array1DInt,
)


class ObjectiveFunctionsConfig(BaseModel):
    """Configuration class for objective functions.

    `ObjectiveFunctionsConfig` defines objective function settings for an
    [`EnOptContext`][ropt.context.EnOptContext] object.

    See the [Configuration guide](../usage/configuration.md#objectives) for
    detailed descriptions and usage examples.

    Attributes:
        weights:             Weights for the objective functions (default: 1.0).
        realization_filters: Optional indices of realization filters.
        function_estimators: Optional indices of function estimators.
        transforms:          Optional indices of objective transforms.
    """

    weights: Array1D = np.array(1.0)
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
    def _broadcast_and_normalize(self) -> Self:
        weights = normalize(self.weights)
        return self.model_copy(
            update={
                "weights": normalize(self.weights),
                "realization_filters": broadcast_1d_array(
                    self.realization_filters, "realization_filters", weights.size
                ),
                "function_estimators": broadcast_1d_array(
                    self.function_estimators, "function_estimators", weights.size
                ),
                "transforms": broadcast_1d_array(
                    self.transforms, "transforms", weights.size
                ),
            }
        )
