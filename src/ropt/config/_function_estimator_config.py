"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FunctionEstimatorConfig(BaseModel):
    """Configuration class for function estimators.

    `FunctionEstimatorConfig` configures a function estimator plugin, which
    controls how objective and constraint function values (and their gradients)
    are combined across realizations.

    See the [Configuration
    guide](../usage/configuration.md#function-estimators) for detailed
    descriptions and usage examples.

    Attributes:
        method:  Name of the function estimator method.
        options: Dictionary of options for the function estimator.
    """

    method: str = "default/default"
    options: dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )
