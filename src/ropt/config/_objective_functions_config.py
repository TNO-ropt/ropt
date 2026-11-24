"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt._utils import (
    broadcast_1d_array,
    broadcast_tuple,
    check_scales,
    normalize,
    update_fields,
)

from ._validated_types import (  # ruff: ignore[typing-only-first-party-import]
    Array1D,
    Array1DBool,
    Indices,
)


class ObjectiveFunctionsConfig(BaseModel):
    """Configuration class for objective functions.

    `ObjectiveFunctionsConfig` defines objective function settings. It is the
    `objectives` section of the configuration.

    See [Configuration Reference](../optimizer_setup/configuration_sections.md#objectives)
    for field descriptions and defaults.

    The reported `target_objective` is divided by the estimated `auto_scale`
    factors, so optimizations that differ in that setting compare only through
    `aggregates.objectives` or their variables.

    Attributes:
        weights:                 Weights for the objective functions (default: 1.0).
        scales:                  Scale factors for the objective functions (default: 1.0).
        offsets:                 Offsets for the objective functions (default: 0.0).
        auto_scale:              Estimate additional scales from the first batch.
        maximize:                Which objectives to maximize (default: `False`).
        realization_filters:     Realization filter index per objective (default: none).
        realization_aggregators: Realization aggregator index per objective (default: `0`).
    """

    weights: Array1D = np.array(1.0)
    scales: Array1D = np.array(1.0)
    offsets: Array1D = np.array(0.0)
    auto_scale: bool = False
    maximize: Array1DBool = np.array(0)
    realization_filters: Indices = (None,)
    realization_aggregators: Indices = (0,)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_normalize(self) -> Self:
        weights = normalize(self.weights)
        update_fields(
            self,
            weights=normalize(self.weights),
            scales=check_scales(self.scales, "scales", weights.size),
            offsets=broadcast_1d_array(self.offsets, "offsets", weights.size),
            maximize=broadcast_1d_array(self.maximize, "maximize", weights.size),
            realization_filters=broadcast_tuple(
                self.realization_filters, "realization_filters", weights.size
            ),
            realization_aggregators=broadcast_tuple(
                self.realization_aggregators, "realization_aggregators", weights.size
            ),
        )
        return self
