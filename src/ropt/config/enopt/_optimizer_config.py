"""Configuration class for the optimizer."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Any, Self

from pydantic import (
    ConfigDict,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

from ropt.config.utils import ImmutableBaseModel


class OptimizerConfig(ImmutableBaseModel):
    """Configuration class for the optimization algorithm.

    This class, `OptimizerConfig`, defines the configuration for the optimization
    algorithm used in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    While optimization methods can have diverse parameters, this class provides a
    standardized set of settings that are commonly used and forwarded to the
    optimizer:

    - **`max_iterations`**: The maximum number of iterations allowed. The
      optimizer may choose to ignore this.
    - **`max_functions`**: The maximum number of function evaluations allowed.
    - **`max_batches`**: The maximum number of evaluations batches allowed. The
      optimizer callback may ask to evaluate a batch of multiple functions and
      gradients at once. This setting will limit the number of those calls.
    - **`tolerance`**: The convergence tolerance used as a stopping criterion.
      The exact definition depends on the optimizer, and it may be ignored.
    - **`speculative`**: If `True`, forces early gradient evaluations, even if
      not strictly required. This can improve load balancing on HPC clusters but
      is only effective if the optimizer supports it. This is disabled if
      `split_evaluations` is `True`.
    - **`split_evaluations`**: If `True`, forces separate function and gradient
      evaluations, even if the optimizer requests them together. This is useful
      with realization filters that completely disable some realizations, to
      potentially reduce the number of evaluations for gradients (see
      [`RealizationFilterConfig`][ropt.config.enopt.RealizationFilterConfig]).
    - **`parallel`**: If `True`, allows the optimizer to use parallelized
      function evaluations. This typically applies to gradient-free methods and
      may be ignored.
    - **`output_dir`**: An optional output directory where the optimizer can
      store files.
    - **`options`**: A dictionary or list of strings for generic optimizer
      options. The required format and interpretation depend on the specific
      optimization method.
    - **`stdout`**: Redirect optimizer standard output to the given file.
    - **`stderr`**: Redirect optimizer standard error to the given file.

    Note: Differences between `max_iterations`, `max_functions`, and `max_batches`
      These three parameters provide different ways to limit the duration or
      computational cost of the optimization process:

      - **`max_iterations`**: This limit is passed directly to the backend
        optimization algorithm. Many optimizers define an "iteration" as a
        distinct step in their process, which might involve one or more
        function or gradient evaluations. The interpretation of `max_iterations`
        depends on the specific backend optimizer; it typically caps the number
        of these internal iterations. Some backends might ignore this setting if
        they don't have a clear concept of iterations.

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
        method:            Name of the optimization method.
        max_iterations:    Maximum number of iterations (optional).
        max_functions:     Maximum number of function evaluations (optional).
        max_batches:       Maximum number of batch evaluations (optional).
        tolerance:         Convergence tolerance (optional).
        speculative:       Force early gradient evaluations (default: `False`).
        split_evaluations: Force separate function/gradient evaluations (default: `False`).
        parallel:          Allow parallelized function evaluations (default: `False`).
        output_dir:        Output directory for the optimizer (optional).
        options:           Generic options for the optimizer (optional).
        stdout:            File to redirect optimizer standard output (optional).
        stderr:            File to redirect optimizer standard error (optional).
    """

    method: str = "scipy/default"
    max_iterations: PositiveInt | None = None
    max_functions: PositiveInt | None = None
    max_batches: PositiveInt | None = None
    tolerance: NonNegativeFloat | None = None
    speculative: bool = False
    split_evaluations: bool = False
    parallel: bool = False
    output_dir: Path | None = None
    options: dict[str, Any] | list[str] | None = None
    stdout: Path | None = None
    stderr: Path | None = None

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
    )

    @model_validator(mode="after")
    def _method(self) -> Self:
        self._mutable()
        plugin, sep, method = self.method.rpartition("/")
        if (sep == "/") and (plugin == "" or method) == "":
            msg = f"malformed method specification: `{self.method}`"
            raise ValueError(msg)
        self._mutable()
        return self
