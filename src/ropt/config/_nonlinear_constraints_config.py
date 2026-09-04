"""Configuration class for non-linear constraints."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt._utils import (
    broadcast_1d_array,
    broadcast_arrays,
    broadcast_keys,
    check_scales,
)

from ._validated_types import (  # ruff: ignore[typing-only-first-party-import]
    Array1D,
    Array1DBool,
    Keys,
)


class NonlinearConstraintsConfig(BaseModel):
    r"""Configuration class for non-linear constraints.

    `NonlinearConstraintsConfig` defines nonlinear constraints used as the
    `nonlinear_constraints` field of an
    [`EnOptContext`][ropt.context.EnOptContext] object.

    See the [Configuration
    guide](../optimizer_setup/configuration.md#nonlinear_constraints) for detailed
    descriptions and usage examples.

    Attributes:
        lower_bounds:        Lower bounds for the right-hand-side values.
        upper_bounds:        Upper bounds for the right-hand-side values.
        scales:              Scale factors for the constraint functions (default: 1.0).
        auto_scale:          Which constraints to estimate an additional scale for,
                             from the first batch (default: `False`).
        realization_filters: Realization filter to apply to each constraint, by key,
                            `None` to apply none (default: `None`).
        function_estimators: Function estimator to apply to each constraint, by key
                             (default: `"0"`).
    """

    lower_bounds: Array1D
    upper_bounds: Array1D
    scales: Array1D = np.array(1.0)
    auto_scale: Array1DBool = np.array(0)
    realization_filters: Keys = (None,)
    function_estimators: Keys = ("0",)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_check(self) -> Self:
        lower_bounds, upper_bounds = broadcast_arrays(
            self.lower_bounds, self.upper_bounds
        )
        return self.model_copy(
            update={
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "scales": check_scales(self.scales, "scales", lower_bounds.size),
                "auto_scale": broadcast_1d_array(
                    self.auto_scale, "auto_scale", lower_bounds.size
                ),
                "realization_filters": broadcast_keys(
                    self.realization_filters, "realization_filters", lower_bounds.size
                ),
                "function_estimators": broadcast_keys(
                    self.function_estimators, "function_estimators", lower_bounds.size
                ),
            }
        )
