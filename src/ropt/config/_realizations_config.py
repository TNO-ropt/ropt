"""Configuration class for realizations."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from ropt._utils import normalize

from ._validated_types import Array1D  # noqa: TC001


class RealizationsConfig(BaseModel):
    """Configuration class for realizations.

    `RealizationsConfig` defines realization ensemble settings for an
    [`EnOptContext`][ropt.context.EnOptContext] object.

    See the [Configuration guide](../usage/configuration.md#realizations) for
    detailed descriptions and usage examples.

    Attributes:
        weights:                 Weights for the realizations (default: 1.0).
        realization_min_success: Minimum number of successful realizations (default:
                                equal to the number of realizations).
    """

    weights: Array1D = np.array(1.0)
    realization_min_success: NonNegativeInt | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _broadcast_normalize_and_check(self) -> Self:
        weights = normalize(self.weights)
        realization_min_success = self.realization_min_success
        if realization_min_success is None or realization_min_success > weights.size:
            realization_min_success = weights.size
        return self.model_copy(
            update={
                "weights": weights,
                "realization_min_success": realization_min_success,
            }
        )
