"""Result objects returned by the high-level API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.enums import ExitCode
    from ropt.results import FunctionResults


@dataclass
class OptimizeResult:
    """The outcome of a single optimization run.

    See [High-Level API](../usage/simple.md) for a walkthrough.

    Attributes:
        exit_code:        The exit code describing how the optimization terminated.
        variables:        The optimal variable vector, or `None` if no valid
                          result was found.
        target_objective: The weighted objective at the optimum, or `None`.
        objectives:       The individual objective values at the optimum, shape
                          `(n_obj,)`, or `None`.
        constraints:      The nonlinear constraint values at the optimum, or `None`.
        results:          The full low-level
                          [`FunctionResults`][ropt.results.FunctionResults] object
                          for the optimum, or `None`.
    """

    exit_code: ExitCode
    variables: NDArray[np.float64] | None
    target_objective: float | None
    objectives: NDArray[np.float64] | None
    constraints: NDArray[np.float64] | None
    results: FunctionResults | None


@dataclass
class EvaluateResult:
    """The outcome of evaluating a single variable vector.

    [`evaluate_many`][ropt.simple.evaluate_many] returns one of these per
    input vector. Each field has the same shape as its counterpart on
    [`OptimizeResult`][ropt.simple.OptimizeResult]. See
    [High-Level API](../usage/simple.md) for a walkthrough.

    Attributes:
        exit_code:        The exit code describing the outcome for this vector.
        target_objective: The weighted objective, or `None` where the evaluation
                          produced no valid result.
        objectives:       The individual objective values, shape `(n_obj,)`, or
                          `None`.
        constraints:      The constraint values, shape `(n_con,)`, or `None` when
                          there are no nonlinear constraints or the evaluation
                          produced no valid result.
        results:          The full low-level
                          [`FunctionResults`][ropt.results.FunctionResults] object.
    """

    exit_code: ExitCode
    target_objective: float | None
    objectives: NDArray[np.float64] | None
    constraints: NDArray[np.float64] | None
    results: FunctionResults
