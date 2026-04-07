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

    `ropt` supports multi-objective optimization. Multiple objectives are
    combined into a single value by summing them after weighting. The `weights`
    field, a `numpy` array, determines the weight of each objective function.
    The length of this array defines the number of objective functions. The
    weights are automatically normalized to sum to 1 (e.g., `[1, 1]` becomes
    `[0.5, 0.5]`).

    Objective functions can optionally be processed using [`realization
    filters`][ropt.realization_filter.RealizationFilter], [`function
    estimators`][ropt.function_estimator.FunctionEstimator], and
    [`transforms`][ropt.transforms.ObjectiveTransform] objects. The
    `realization_filters`, `function_estimators`, and `transforms` fields are
    integer index arrays: each entry selects an object by its position in the
    corresponding tuple defined in [`EnOptContext`][ropt.context.EnOptContext].
    An out-of-range index means no object is applied to that objective. If a
    field uses its default value of `-1`, no object is applied at all.

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
