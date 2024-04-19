"""Configuration class for the backend optimizer."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import Any, Dict, List, Optional, Union

from pydantic import NonNegativeFloat, PositiveInt  # noqa: TCH002

from ._enopt_base_model import EnOptBaseModel
from .constants import DEFAULT_OPTIMIZATION_BACKEND


class OptimizerConfig(EnOptBaseModel):
    """The configuration class for optimizers used in the optimization.

    This class defines the configuration for optimizers, configured by the
    `optimizer` field in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig]
    object.

    To perform the optimization, low-level optimization algorithms are employed,
    implemented in optimizer backends. The `backend` field is used to select the
    backend, which may be either built-in (such as the default `SciPy` backend)
    or installed separately as a plugin. A backend may implement multiple
    algorithms, and the `algorithm` field generally determines which one will be
    used.

    Although there may be significant differences in the parameters that can be
    used for different algorithms and backends, there are a few standard settings
    defined in this configuration object, which are forwarded to the backend:

    - The maximum number of iterations allowed before the optimization should be
      aborted by this optimizer. A backend may choose to ignore this option.
    - The maximum number of function evaluations allowed before the optimization
      is aborted.
    - The convergence tolerance used as a stopping criterion. The exact
      definition of the criterion depends on the backend. A backend may choose to
      ignore this option.
    - Whether gradients should be evaluated early, even if the optimizer does not
      strictly need it yet. When evaluating on a distributed HPC cluster, this
      may lead to better load-balancing for some backends/algorithms. This option
      is only applied if the backend optimization algorithm knows how to make use
      of it.
    - Whether calculations for functions and gradients should be done separately,
      even if the optimizer requests them to be evaluated together. This option
      is useful when a filter is specified that deactivates some realizations
      (see
      [`RealizationFilterConfig`][ropt.config.enopt.RealizationFilterConfig]). In
      this case, after evaluation of the functions, it may be possible to reduce
      the number of evaluations for a following gradient calculation.
    - Whether the algorithm may use parallelized function evaluations. This
      option currently only applies to gradient-free algorithms and may be
      ignored by the backend.
    - An optional location of an output directory, where the backend algorithm
      may store files.
    - Generic backend algorithm options to be interpreted by the backend. The
      options may be passed as an arbitrary dictionary, or as a list of strings.
      It depends on the backend what form is required and how it is interpreted.

    Attributes:
        backend:           The name of the optimization backend
        algorithm:         Optional name of the optimization algorithm used
        max_iterations:    Optional maximum number of iterations
        max_functions:     Optional maximum number of function evaluations
        tolerance:         Optional tolerance for convergence
        speculative:       Force gradient evaluations; disabled if
                          split_evaluations is True (default `False`)
        split_evaluations: Evaluate function and gradient separately
                          (default: `False`)
        parallel:          Allow for parallelized evaluation (default: `False`)
        output_dir:        Optional output directory for use by the backend
        options:           Optional generic options for use by the backend
    """

    backend: str = DEFAULT_OPTIMIZATION_BACKEND
    algorithm: Optional[str] = None
    max_iterations: Optional[PositiveInt] = None
    max_functions: Optional[PositiveInt] = None
    tolerance: Optional[NonNegativeFloat] = None
    speculative: bool = False
    split_evaluations: bool = False
    parallel: bool = False
    output_dir: Optional[Path] = None
    options: Optional[Union[Dict[str, Any], List[str]]] = None
