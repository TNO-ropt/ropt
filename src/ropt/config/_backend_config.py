"""Configuration class for the optimizer."""

from __future__ import annotations

from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)


class BackendConfig(BaseModel):
    """Configuration class for the optimization backend.

    `BackendConfig` defines the configuration for the optimization algorithm
    used by an optimization backend plugin.

    See the [Configuration guide](../usage/configuration.md#backend) for
    detailed descriptions and usage examples.

    Attributes:
        method:                Name of the optimization method.
        max_iterations:        Maximum number of iterations (optional).
        convergence_tolerance: Convergence tolerance (optional).
        parallel:              Allow parallelized function evaluations (default: `False`).
        options:               Generic options for the optimizer (optional).
    """

    method: str = "scipy/default"
    max_iterations: PositiveInt | None = None
    convergence_tolerance: NonNegativeFloat | None = None
    parallel: bool = False
    options: dict[str, Any] | list[str] | None = None

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _method(self) -> Self:
        plugin, sep, method = self.method.rpartition("/")
        if (sep == "/") and not (not plugin or method):
            msg = f"malformed method specification: `{self.method}`"
            raise ValueError(msg)
        return self
