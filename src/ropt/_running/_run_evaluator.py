"""Evaluate a batch and emit its lifecycle events."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._core import EnsembleEvaluator
from ropt._logging import get_logger
from ropt._scaling import scale
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import FunctionResults

from ._events import (
    emit_event,
    emit_finished_evaluation,
    emit_start_evaluation,
    validate_handlers,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from ropt._evaluation import EvaluationBatchCallback
    from ropt._handlers import EventHandler
    from ropt.context import EnOptContext

_logger = get_logger(__name__)


def run_evaluation(
    context: EnOptContext,
    variables: ArrayLike,
    evaluator: EvaluationBatchCallback,
    handlers: Sequence[EventHandler] = (),
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Evaluate variable vectors, delivering events to handlers in order.

    Args:
        context:   The optimization context.
        variables: A vector or matrix of variable vectors.
        evaluator: The callable that evaluates batches.
        handlers:  Event handlers in delivery order.
        metadata:  Default metadata passed to evaluations and attached to results.

    Raises:
        ValueError: If the input variables have the wrong shape.
    """
    validate_handlers(handlers)
    variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=2)
    if variables.shape[-1] != context.variables.variable_count:
        msg = "The input variables have the wrong shape"
        raise ValueError(msg)
    context.lock()

    def stop() -> None:
        pass

    default_metadata = {} if metadata is None else deepcopy(metadata)

    _logger.info("Starting evaluation")
    emit_event(
        EnOptEvent(event_type=EnOptEventType.START_ENSEMBLE_EVALUATOR, context=context),
        handlers,
        stop,
    )
    variables = scale(variables, context.variables.scales, context.variables.offsets)
    ensemble_evaluator = EnsembleEvaluator(
        context,
        evaluator,
        signal_start=partial(
            emit_start_evaluation, context, handlers, stop, default_metadata
        ),
        signal_finished=partial(emit_finished_evaluation, context, handlers, stop),
    )
    results = ensemble_evaluator.calculate(
        variables, compute_functions=True, compute_gradients=False
    )
    assert results
    assert isinstance(results[0], FunctionResults)
    _logger.info("Evaluation finished")
    emit_event(
        EnOptEvent(
            event_type=EnOptEventType.FINISHED_ENSEMBLE_EVALUATOR,
            context=context,
            results=results,
        ),
        handlers,
        stop,
    )
