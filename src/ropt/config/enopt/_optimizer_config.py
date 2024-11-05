"""Configuration class for the optimizer."""

from __future__ import annotations

import sys
from pathlib import Path  # noqa: TCH003
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    ConfigDict,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

from ropt.config.utils import ImmutableBaseModel

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class OptimizerConfig(ImmutableBaseModel):
    """The configuration class for optimizers used in the optimization.

    This class defines the configuration for optimizers, configured by the
    `optimizer` field in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig]
    object.

    Although there may be significant differences in the parameters that can be
    used for different optimization methods, there are a few standard settings
    defined in this configuration object, which are forwarded to the optimizer:

    - The maximum number of iterations allowed before the optimization should be
      aborted by this optimizer. The optimizer may choose to ignore this option.
    - The maximum number of function evaluations allowed before the optimization
      is aborted.
    - The convergence tolerance used as a stopping criterion. The exact
      definition of the criterion depends on the optimizer. The optimizer may
      choose to ignore this option.
    - Whether gradients should be evaluated early, even if the optimizer does
      not strictly need it yet. When evaluating on a distributed HPC cluster,
      this may lead to better load-balancing for some methods. This option is
      only applied if the optimization algorithm knows how to make use of it.
    - Whether calculations for functions and gradients should be done
      separately, even if the optimizer requests them to be evaluated together.
      This option is useful when a filter is specified that deactivates some
      realizations (see
      [`RealizationFilterConfig`][ropt.config.enopt.RealizationFilterConfig]).
      In this case, after evaluation of the functions, it may be possible to
      reduce the number of evaluations for a following gradient calculation.
    - Whether the optimizer may use parallelized function evaluations. This
      option currently only applies to gradient-free methods and may be ignored
      by the optimizer.
    - An optional location of an output directory, where the optimizer may store
      files.
    - Generic optimizer options that may be passed as an arbitrary dictionary,
      or as a list of strings. It depends on the method what form is required
      and how it is interpreted.

    Attributes:
        method:            Name of the optimization method used.
        max_iterations:    Optional maximum number of iterations.
        max_functions:     Optional maximum number of function evaluations.
        tolerance:         Optional tolerance for convergence.
        speculative:       Force gradient evaluations; disabled if
                           split_evaluations is True (default `False`).
        split_evaluations: Evaluate function and gradient separately
                           (default: `False`).
        parallel:          Allow for parallelized evaluation (default: `False`).
        output_dir:        Optional output directory for use by the optimizer.
        options:           Optional generic options for use by the optimizer.
    """

    method: str = "scipy/default"
    max_iterations: Optional[PositiveInt] = None
    max_functions: Optional[PositiveInt] = None
    tolerance: Optional[NonNegativeFloat] = None
    speculative: bool = False
    split_evaluations: bool = False
    parallel: bool = False
    output_dir: Optional[Path] = None
    options: Optional[Union[Dict[str, Any], List[str]]] = None

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
