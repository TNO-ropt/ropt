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
    """The configuration class for objective functions.

    This configuration class defines objective functions configured by the
    `objectives` field in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig]
    object.

    `ropt` supports optimization over multiple objectives, which are summed after
    weighting with values passed via the `weights` field. This field is a
    `numpy` array, with a length that determines the number of objective functions.
    Its values will be normalized to have a sum equal to 1. For example, when
    `weights` is set to `[1, 1]`, the stored values will be `[0.5, 0.5]`.

    The objective functions may be subject to realization filters and function
    estimators. The `realization_filters` and `function_estimators` fields
    contain indices to the realization filter or function estimator objects to
    use. The objects referred to are configured in the parent
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Attributes:
        weights:             Objective functions weights (default: 1.0).
        realization_filters: Optional realization filter indices.
        function_estimators: Optional function estimator indices.
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
