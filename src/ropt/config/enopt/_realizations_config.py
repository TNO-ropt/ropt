"""Configuration class for realizations."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, NonNegativeInt, model_validator

from ropt.config.utils import ImmutableBaseModel, normalize
from ropt.config.validated_types import Array1D  # noqa: TC001


class RealizationsConfig(ImmutableBaseModel):
    """Configuration class for realizations.

    This class, `RealizationsConfig`, defines the configuration for realizations
    used in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    To optimize an ensemble of functions, a set of realizations is defined. When
    the optimizer requests a function value or a gradient, these are calculated for
    each realization and then combined into a single value. Typically, this
    combination is a weighted sum, but other methods are possible.

    The `weights` field, a `numpy` array, determines the weight of each
    realization. The length of this array defines the number of realizations. The
    weights are automatically normalized to sum to 1 (e.g., `[1, 1]` becomes
    `[0.5, 0.5]`).

    If function value calculations for some realizations fail (e.g., due to a
    simulation error), the total function and gradient values can still be
    calculated by excluding the missing values. However, a minimum number of
    successful realizations may be required. The `realization_min_success` field
    specifies this minimum. By default, it is set equal to the number of
    realizations, meaning no missing values are allowed.

    Note:
        Setting `realization_min_success` to zero allows the optimization to
        proceed even if all realizations fail. While some optimizers can handle
        this, most will treat it as if the value were one, requiring at least
        one successful realization.

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
    )

    @model_validator(mode="after")
    def _broadcast_normalize_and_check(self) -> Self:
        self._mutable()
        self.weights = normalize(self.weights)
        if (
            self.realization_min_success is None
            or self.realization_min_success > self.weights.size
        ):
            self.realization_min_success = self.weights.size
        self._immutable()
        return self
