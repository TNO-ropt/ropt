"""Configuration class for the optimizer."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
)


class OptimizerConfig(BaseModel):
    """Configuration class for the optimization algorithm.

    This class, `BackendConfig`, defines the configuration for the optimization
    algorithm used in an [`EnOptConfig`][ropt.config.EnOptConfig] object.

    While optimization methods can have diverse parameters, this class provides a
    standardized set of settings that are commonly used and forwarded to the
    optimizer:

      - **`max_batches`**: This limit restricts the total number of *calls* made
        to the evaluation function provided to `ropt`. An optimizer might request
        a batch containing multiple function and/or gradient evaluations within
        a single call. `max_batches` limits how many such batch requests are
        processed sequentially. This is particularly useful for managing resource
        usage when batches are evaluated in parallel (e.g., on an HPC cluster),
        as it controls the number of sequential submission steps. The number of
        batches does not necessarily correspond directly to the number of
        optimizer iterations, especially if function and gradient evaluations
        occur in separate batches.

      - **`max_functions`**: This imposes a hard limit on the total *number* of
        individual objective function evaluations performed across all batches.
        Since a single batch evaluation (limited by `max_batches`) can involve
        multiple function evaluations, setting `max_functions` provides more
        granular control over the total computational effort spent on function
        calls. It can serve as an alternative stopping criterion if the backend
        optimizer doesn't support `max_iterations` or if you need to strictly
        limit the function evaluation count. Note that exceeding this limit might
        cause the optimization to terminate mid-batch, potentially earlier than
        a corresponding `max_batches` limit would.

    Attributes:
        max_functions:  Maximum number of function evaluations (optional).
        max_batches:    Maximum number of batch evaluations (optional).
    """

    max_functions: PositiveInt | None = None
    max_batches: PositiveInt | None = None

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )
