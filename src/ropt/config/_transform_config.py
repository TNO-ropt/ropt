"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ._validated_types import (  # ruff: ignore[typing-only-first-party-import]
    Array1DBool,
)


class VariableTransformConfig(BaseModel):
    """Configuration class for variable transforms.

    `VariableTransformConfig` configures a
    [`VariableTransform`][ropt.transforms.VariableTransform] plugin that
    transforms variables to the optimizer's domain.

    See the [Configuration guide](../optimizer_setup/configuration.md#transforms) for
    detailed descriptions and usage examples.

    Attributes:
        method:  Name of the variable transform method.
        options: Dictionary of options for the variable transform method.
        mask:    Optional boolean array selecting the variables this transform
                 applies to (default: all).
    """

    method: str = "default/default"
    options: dict[str, Any] = {}
    mask: Array1DBool | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )
