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

    This class, `BackendConfig`, defines the configuration for the optimization
    algorithms implemented in the optimization backends.

    While optimization methods can have diverse parameters, this class provides a
    standardized set of settings that are commonly used and forwarded to the
    optimizer backend:

    - **`max_iterations`**: The maximum number of iterations allowed. The
      optimizer may choose to ignore this.
      optimizer callback may ask to evaluate a batch of multiple functions and
      gradients at once. This setting will limit the number of those calls.
    - **`convergence_tolerance`**: The convergence tolerance used as a stopping
      criterion.
      The exact definition depends on the optimizer, and it may be ignored.
    - **`parallel`**: If `True`, allows the optimizer to use parallelized
      function evaluations. This typically applies to gradient-free methods and
      may be ignored.
    - **`options`**: A dictionary or list of strings for generic optimizer
      options. The required format and interpretation depend on the specific
      optimization method.

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
