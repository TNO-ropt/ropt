"""Configuration class for realization filters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RealizationFilterConfig(BaseModel):
    """Configuration class for realization filters.

    `RealizationFilterConfig` configures a
    [`RealizationFilter`][ropt.realization_filter.RealizationFilter] plugin that
    adjusts per-realization weights.

    See the [Configuration
    guide](../usage/configuration.md#realization-filters) for detailed
    descriptions and usage examples.

    Attributes:
        method:  Name of the realization filter method.
        options: Dictionary of options for the realization filter.
    """

    method: str
    options: dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )
