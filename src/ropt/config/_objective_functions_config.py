"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt._utils import (
    broadcast_1d_array,
    broadcast_keys,
    check_scales,
    normalize,
)

from ._validated_types import (  # ruff: ignore[typing-only-first-party-import]
    Array1D,
    Array1DBool,
    Keys,
)


class ObjectiveFunctionsConfig(BaseModel):
    """Configuration class for objective functions.

    `ObjectiveFunctionsConfig` defines objective function settings for an
    [`EnOptContext`][ropt.context.EnOptContext] object.

    See the [Configuration guide](../optimizer_setup/configuration.md#objectives) for
    detailed descriptions and usage examples.

    Attributes:
        weights:             Weights for the objective functions (default: 1.0).
        scales:              Scale factors for the objective functions (default: 1.0).
        auto_scale:          Estimate additional scales from the first batch.
        maximize:            Which objectives to maximize (default: `False`).
        realization_filters: Realization filter to apply to each objective, by key,
                            `None` to apply none (default: `None`).
        function_estimators: Function estimator to apply to each objective, by key
                             (default: `"0"`).
    """

    weights: Array1D = np.array(1.0)
    scales: Array1D = np.array(1.0)
    auto_scale: bool = False
    maximize: Array1DBool = np.array(0)
    realization_filters: Keys = (None,)
    function_estimators: Keys = ("0",)

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
                "scales": check_scales(self.scales, "scales", weights.size),
                "maximize": broadcast_1d_array(self.maximize, "maximize", weights.size),
                "realization_filters": broadcast_keys(
                    self.realization_filters, "realization_filters", weights.size
                ),
                "function_estimators": broadcast_keys(
                    self.function_estimators, "function_estimators", weights.size
                ),
            }
        )
