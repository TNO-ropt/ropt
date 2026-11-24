"""Configuration class for realization aggregators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RealizationAggregatorConfig(BaseModel):
    """Configuration class for realization aggregators.

    `RealizationAggregatorConfig` configures a realization aggregator plugin,
    which controls how objective and constraint function values (and their
    gradients) are combined across realizations.

    See [Configuration
    Sections](../optimizer_setup/configuration_sections.md#realization-aggregator)
    for detailed descriptions and usage examples.

    Attributes:
        method:  Name of the realization aggregator method.
        options: Dictionary of options for the realization aggregator.
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
