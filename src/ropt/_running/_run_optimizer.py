"""Run an optimization and emit its lifecycle events."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._core import EnsembleEvaluator, EnsembleOptimizer
from ropt._logging import get_logger
from ropt._scaling import scale
from ropt.enums import EnOptEventType, ExitCode
from ropt.events import EnOptEvent
from ropt.exceptions import AbortedError, OptimizerStop

from ._events import (
    emit_event,
    emit_finished_evaluation,
    emit_start_evaluation,
    validate_handlers,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from ropt._concurrency import AbortScope
    from ropt._evaluation import EvaluationBatchCallback
    from ropt._handlers import EventHandler
    from ropt.context import EnOptContext
    from ropt.results import FunctionResults, GradientResults, Results

_logger = get_logger(__name__)


def run_optimization(  # ruff: ignore[too-many-arguments]
    context: EnOptContext,
    variables: ArrayLike,
    evaluator: EvaluationBatchCallback,
    handlers: Sequence[EventHandler] = (),
    *,
    abort_scope: AbortScope | None = None,
    metadata: dict[str, Any] | None = None,
    f0: FunctionResults | None = None,
    g0: GradientResults | None = None,
) -> ExitCode:
    """Run an optimization, delivering events to handlers in order.

    Args:
        context:     The optimization context.
        variables:   The initial variable vector.
        evaluator:   The callable that evaluates batches.
        handlers:    Event handlers in delivery order.
        abort_scope: An optional scope that cuts this run off.
        metadata:    Default metadata passed to evaluations and attached to results.
        f0:          Optional function results at the initial variables.
        g0:          Optional gradient results at the initial variables.

    Returns:
        The optimization exit code.

    Raises:
        ValueError: If the input variables have the wrong shape.
    """
    validate_handlers(handlers)
    variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)
    if variables.shape != (context.variables.variable_count,):
        msg = "The input variables have the wrong shape"
        raise ValueError(msg)
    context.lock()
    stop_requested = False

    def stop() -> None:
        nonlocal stop_requested
        stop_requested = True

    default_metadata = None if metadata is None else deepcopy(metadata)

    def check_stop() -> None:
        if stop_requested:
            raise OptimizerStop(ExitCode.STOPPED)
        if abort_scope is not None and abort_scope.aborting:
            raise AbortedError(abort_scope.exit_code)

    def start_evaluation() -> dict[str, object] | None:
        batch_metadata = emit_start_evaluation(
            context, handlers, stop, default_metadata
        )
        check_stop()
        return batch_metadata

    def finish_evaluation(results: tuple[Results, ...]) -> None:
        emit_finished_evaluation(context, handlers, stop, results)
        check_stop()

    _logger.debug(
        "Configuration: %d variable(s), %d realization(s), %d objective(s)",
        context.variables.variable_count,
        context.realizations.weights.size,
        context.objectives.weights.size,
    )
    _logger.info("Starting optimization")
    emit_event(
        EnOptEvent(event_type=EnOptEventType.START_OPTIMIZER, context=context),
        handlers,
        stop,
    )

    variables = scale(variables, context.variables.scales, context.variables.offsets)
    ensemble_evaluator = EnsembleEvaluator(
        context,
        evaluator,
        f0=f0,
        g0=g0,
        signal_start=start_evaluation,
        signal_finished=finish_evaluation,
    )
    ensemble_optimizer = EnsembleOptimizer(
        context=context,
        ensemble_evaluator=ensemble_evaluator,
    )
    exit_code = ensemble_optimizer.start(variables)

    _logger.info("Optimization finished: %s", exit_code.name)
    emit_event(
        EnOptEvent(event_type=EnOptEventType.FINISHED_OPTIMIZER, context=context),
        handlers,
        stop,
    )
    return exit_code
