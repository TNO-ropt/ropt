"""The high-level ``optimize`` entry point."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ropt.context import EnOptContext
from ropt.workflow.compute_steps import OptimizationStep
from ropt.workflow.event_handlers import ResultsHandler

from ._objective import adapt_objective
from ._result import OptimizeResult
from ._session import current_session, make_evaluator, run_step

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.enums import ExitCode
    from ropt.results import FunctionResults
    from ropt.workflow.event_handlers import EventHandler

    from ._objective import ObjectiveCallback
    from ._session import Session


def optimize(
    config: dict[str, Any],
    x0: ArrayLike,
    objective: ObjectiveCallback,
    *,
    handlers: Sequence[EventHandler] | None = None,
    constraint_tolerance: float = 1e-10,
) -> OptimizeResult:
    """Run a single optimization.

    See [High-Level API](../usage/highlevel.md) for a walkthrough.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        objective:            The per-realization objective callback.
        handlers:             Optional local result handlers, each owned by this
                              optimization alone.
        constraint_tolerance: The tolerance within which a constraint is
                              considered satisfied.

    Returns:
        A [`OptimizeResult`][ropt.highlevel.OptimizeResult] describing the outcome.
    """
    return _optimize(
        current_session(),
        config,
        x0,
        objective,
        handlers=handlers,
        constraint_tolerance=constraint_tolerance,
    )


def _optimize(  # ruff: ignore[too-many-arguments]
    session: Session | None,
    config: dict[str, Any],
    x0: ArrayLike,
    objective: ObjectiveCallback,
    *,
    handlers: Sequence[EventHandler] | None,
    constraint_tolerance: float,
) -> OptimizeResult:
    context = EnOptContext.model_validate(config)
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )

    evaluator = make_evaluator(adapt_objective(objective, n_obj, n_con), session)
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    step = OptimizationStep(evaluator=evaluator)
    step.add_event_handler(result_handler)
    for handler in handlers or ():
        handler.claim()
        step.add_event_handler(handler)

    exit_code = run_step(
        step, context=context, variables=np.asarray(x0, dtype=np.float64)
    )
    return _build_run_result(exit_code, result_handler["results"])


def _build_run_result(
    exit_code: ExitCode, results: FunctionResults | None
) -> OptimizeResult:
    if results is None or results.functions is None:
        return OptimizeResult(
            exit_code=exit_code,
            variables=None,
            target_objective=None,
            objectives=None,
            constraints=None,
            results=None,
        )
    return OptimizeResult(
        exit_code=exit_code,
        variables=results.evaluations.variables,
        target_objective=float(results.functions.target_objective),
        objectives=results.functions.objectives,
        constraints=results.functions.constraints,
        results=results,
    )


def optimize_many(
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    objective: ObjectiveCallback | Sequence[ObjectiveCallback],
    *,
    limit: int | None = None,
    constraint_tolerance: float = 1e-10,
) -> tuple[OptimizeResult, ...]:
    """Run several optimizations concurrently, sharing the open session.

    Each of `config`, `x0`, and `objective` may be a single value (used for
    every run) or a sequence (one per run). Sequences set the number of runs and
    must agree in length; single values are broadcast. A single `x0` is a 1-D
    vector; a per-run sequence of `x0`s is a 2-D matrix with one vector per row.

    The runs execute concurrently on driver threads and share the session's
    worker pool, so this must be called inside a `threads`/`processes` block.
    `limit` bounds how many run simultaneously. The first run to raise cancels
    the rest and propagates (fail-fast). See
    [High-Level API](../usage/highlevel.md) for a walkthrough.

    Args:
        config:               The configuration, or one per run.
        x0:                   The initial variable vector, or one per row.
        objective:            The objective callback, or one per run.
        limit:                The maximum number of runs to execute at once.
        constraint_tolerance: The tolerance within which a constraint is
                              considered satisfied.

    Returns:
        One [`OptimizeResult`][ropt.highlevel.OptimizeResult] per run, in order.

    Raises:
        RuntimeError: If no `threads`/`processes` block is open.
    """
    session = current_session()
    if session is None:
        msg = (
            "optimize_many() requires an execution block, "
            "e.g. `with ropt.threads(...):`."
        )
        raise RuntimeError(msg)

    runs = _broadcast(config, x0, objective)
    jobs: list[Callable[[], OptimizeResult]] = [
        partial(
            _optimize,
            session,
            run_config,
            run_x0,
            run_objective,
            handlers=None,
            constraint_tolerance=constraint_tolerance,
        )
        for run_config, run_x0, run_objective in runs
    ]
    return tuple(session.gather(jobs, limit))


def _broadcast(
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    objective: ObjectiveCallback | Sequence[ObjectiveCallback],
) -> list[tuple[dict[str, Any], ArrayLike, ObjectiveCallback]]:
    configs = [config] if isinstance(config, Mapping) else list(config)
    objectives = [objective] if callable(objective) else list(objective)

    x0_array = np.asarray(x0, dtype=np.float64)
    if x0_array.ndim == 1:
        x0s: list[ArrayLike] = [x0_array]
    elif x0_array.ndim == 2:  # ruff: ignore[magic-value-comparison]
        x0s = list(x0_array)
    else:
        msg = "x0 must be a vector or a 2-D matrix of vectors."
        raise ValueError(msg)

    counts = {len(seq) for seq in (configs, objectives, x0s) if len(seq) != 1}
    if len(counts) > 1:
        msg = "config, x0 and objective sequences must have the same length."
        raise ValueError(msg)
    count = counts.pop() if counts else 1

    def _broadcast_seq(seq: list[Any]) -> list[Any]:
        return seq * count if len(seq) == 1 else seq

    return list(
        zip(
            _broadcast_seq(configs),
            _broadcast_seq(x0s),
            _broadcast_seq(objectives),
            strict=True,
        )
    )
