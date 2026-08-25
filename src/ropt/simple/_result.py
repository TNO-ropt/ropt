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
class EvaluateResult:
    """The values one variable vector produced.

    Returned by [`evaluate`][ropt.simple.evaluate], once per input vector by
    [`evaluate_many`][ropt.simple.evaluate_many], and handed to a `report`
    callback once per function evaluation of a run. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Attributes:
        variables:        The variable vector these values belong to, or `None`.
        target_objective: The weighted objective, or `None` where the evaluation
                          produced no valid result.
        objectives:       The individual objective values, shape `(n_obj,)`, or
                          `None`.
        constraints:      The constraint values, shape `(n_con,)`, or `None` when
                          there are no nonlinear constraints or the evaluation
                          produced no valid result.
        results:          The full low-level
                          [`FunctionResults`][ropt.results.FunctionResults]
                          object, or `None`. Only an optimization that reached
                          no valid result leaves this empty; an evaluation and a
                          `report` callback always have one.
    """

    variables: NDArray[np.float64] | None
    target_objective: float | None
    objectives: NDArray[np.float64] | None
    constraints: NDArray[np.float64] | None
    results: FunctionResults | None


@dataclass
class OptimizeResult(EvaluateResult):
    """The outcome of a single optimization run.

    An optimization ends at one evaluation, the best one it found, so this is
    that evaluation's [`EvaluateResult`][ropt.simple.EvaluateResult] together
    with the exit code of the run that reached it. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Attributes:
        variables:        The optimal variable vector, or `None` if no valid
                          result was found.
        target_objective: The weighted objective at the optimum, or `None`.
        objectives:       The individual objective values at the optimum, shape
                          `(n_obj,)`, or `None`.
        constraints:      The nonlinear constraint values at the optimum, or `None`.
        results:          The full low-level
                          [`FunctionResults`][ropt.results.FunctionResults] object
                          for the optimum, or `None`.
        exit_code:        The exit code describing how the optimization terminated.
    """

    exit_code: ExitCode


def _build_evaluate_result(result: FunctionResults) -> EvaluateResult:
    if result.functions is None:
        return EvaluateResult(
            variables=result.evaluations.variables,
            target_objective=None,
            objectives=None,
            constraints=None,
            results=result,
        )
    return EvaluateResult(
        variables=result.evaluations.variables,
        target_objective=float(result.functions.target_objective),
        objectives=result.functions.objectives,
        constraints=result.functions.constraints,
        results=result,
    )
