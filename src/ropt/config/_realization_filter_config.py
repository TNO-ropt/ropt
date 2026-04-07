"""Configuration class for realization filters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RealizationFilterConfig(BaseModel):
    """Configuration class for realization filters.

    `RealizationFilterConfig` configures a
    [`RealizationFilter`][ropt.realization_filter.RealizationFilter] plugin that
    adjusts per-realization weights. Realization filters are configured as a
    tuple in the `realization_filters` field of
    [`EnOptContext`][ropt.context.EnOptContext]. Objectives and constraints
    reference a specific filter by its index in that tuple.

    By default, objective and constraint functions, as well as their gradients,
    are calculated as a weighted function of all realizations. Realization
    filters provide a way to modify the weights of individual realizations. For
    example, they can be used to select a subset of realizations for calculating
    the final objective and constraint functions and their gradients by setting
    the weights of the other realizations to zero.

    The `method` field specifies the realization filter method to use for
    adjusting the weights. The `options` field allows passing a dictionary of
    key-value pairs to further configure the chosen method. The interpretation
    of these options depends on the selected method.

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
