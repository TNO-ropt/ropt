"""Configuration class for samplers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SamplerConfig(BaseModel):
    """Configuration class for samplers.

    `SamplerConfig` configures a [`Sampler`][ropt.sampler.Sampler] plugin that
    generates variable perturbations for gradient estimation. Samplers are
    configured as a tuple in the `samplers` field of
    [`EnOptContext`][ropt.context.EnOptContext]. Variables reference a specific
    sampler by its index in that tuple via their `samplers` field.

    Samplers generate perturbations added to variables for gradient
    calculations. These perturbations can be deterministic or stochastic.

    The `method` field specifies the sampler method to use for generating
    perturbations. The `options` field allows passing a dictionary of key-value
    pairs to further configure the chosen method. The interpretation of these
    options depends on the selected method.

    By default, each realization uses a different set of perturbed variables.
    Setting the `shared` flag to `True` directs the sampler to use the same set
    of perturbed values for all realizations.

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
