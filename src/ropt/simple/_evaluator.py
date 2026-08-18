"""Build and run the low-level machinery behind a single high-level run."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ropt.components.evaluators import FunctionEvaluator, ParallelEvaluator
from ropt.context import EnOptContext

from ._function import adapt_function

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.compute_steps import ComputeStep
    from ropt.components.evaluators import Evaluator, NameCallback
    from ropt.components.executors import Executor
    from ropt.enums import ExitCode

    from ._function import EvaluationFunction


def make_evaluator(
    config: dict[str, Any],
    function: EvaluationFunction,
    executor: Executor | None,
    get_name: NameCallback | None,
) -> tuple[EnOptContext, Evaluator]:
    """Validate a configuration and wire an evaluator for it.

    The number of objectives and constraints the evaluation function must
    produce follows from the configuration, so the two are built together. With
    an executor the evaluations are spread over its workers, without one they
    run in-process on the calling thread.

    Args:
        config:   The optimization configuration.
        function: The user-supplied evaluation function.
        executor: The active executor, or `None` on the sequential floor.
        get_name: An optional naming callback for the run's tasks.

    Returns:
        The validated context and the evaluator to run it with.
    """
    context = EnOptContext.model_validate(config)
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )
    callback = adapt_function(function, n_obj, n_con)
    if executor is None:
        return context, FunctionEvaluator(function=callback)
    return context, ParallelEvaluator(
        function=callback, executor=executor, get_name=get_name
    )


def run_step(
    step: ComputeStep,
    *,
    context: EnOptContext,
    variables: ArrayLike,
    metadata: dict[str, Any] | None = None,
) -> ExitCode:
    """Run a compute step on the calling thread.

    The step's evaluator is already wired to the session's executor (if any) at
    construction time, so running the step needs no session.

    Args:
        step:      The compute step to run.
        context:   The optimizer context.
        variables: The initial variable vector(s).
        metadata:  Optional dictionary attached to every emitted
                   [`Results`][ropt.results.Results].

    Returns:
        The exit code returned by the step.
    """
    return cast(
        "ExitCode",
        step.run(context=context, variables=variables, metadata=metadata),
    )
