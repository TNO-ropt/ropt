"""Configuration class for the optimizer."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from pydantic import BaseModel, ConfigDict, PositiveInt


class OptimizerConfig(BaseModel):
    """Configuration class for the optimization algorithm.

    `OptimizerConfig` defines workflow-level settings for an optimization run,
    configured as the `optimizer` field of
    [`EnOptContext`][ropt.context.EnOptContext].

    See the [Configuration guide](../usage/configuration.md#optimizer) for
    detailed descriptions and usage examples.

    Attributes:
        max_batches:    Maximum number of batch evaluations (optional).
        max_functions:  Maximum number of function evaluations (optional).
        output_dir:            Output directory for the optimizer (optional).
        stdout:                File to redirect optimizer standard output (optional).
        stderr:                File to redirect optimizer standard error (optional).
    """

    max_batches: PositiveInt | None = None
    max_functions: PositiveInt | None = None
    output_dir: Path | None = None
    stdout: Path | None = None
    stderr: Path | None = None

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )
