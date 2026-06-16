"""Configuration class for samplers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SamplerConfig(BaseModel):
    """Configuration class for samplers.

    `SamplerConfig` configures a [`Sampler`][ropt.sampler.Sampler] plugin that
    generates variable perturbations for gradient estimation.

    See the [Configuration guide](../usage/configuration.md#samplers) for
    detailed descriptions and usage examples.

    Attributes:
        method:  Name of the sampler method.
        options: Dictionary of options for the sampler.
        shared:  Whether to share perturbation values between realizations (default: `False`).
    """

    method: str = "scipy/default"
    options: dict[str, Any] = {}
    shared: bool = False

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )
