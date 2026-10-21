"""Evaluate a batch and emit its lifecycle events."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._core import EnsembleEvaluator
from ropt._logging import get_logger
from ropt._scaling import scale
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import FunctionResults

from ._events import emit_event, validate_handlers

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from ropt._evaluation import EvaluationBatchCallback
    from ropt._handlers import EventHandler
    from ropt.context import EnOptContext
    from ropt.results import Results

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
        metadata:  Optional metadata attached to emitted results.

    Raises:
        ValueError: If the input variables have the wrong shape.
    """
    validate_handlers(handlers)
    variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=2)
    if variables.shape[-1] != context.variables.variable_count:
        msg = "The input variables have the wrong shape"
        raise ValueError(msg)
    context.lock()

    def send(event: EnOptEvent) -> None:
        emit_event(event, handlers, stop)

    def stop() -> None:
        pass

    def signal_evaluation(results: tuple[Results, ...] | None = None) -> None:
        if results is None:
            send(
                EnOptEvent(event_type=EnOptEventType.START_EVALUATION, context=context)
            )
        else:
            if metadata is not None:
                for item in results:
                    item.metadata = deepcopy(metadata)
            send(
                EnOptEvent(
                    event_type=EnOptEventType.FINISHED_EVALUATION,
                    context=context,
                    results=results,
                )
            )

    _logger.info("Starting evaluation")
    send(
        EnOptEvent(event_type=EnOptEventType.START_ENSEMBLE_EVALUATOR, context=context)
    )
    variables = scale(variables, context.variables.scales, context.variables.offsets)
    ensemble_evaluator = EnsembleEvaluator(
        context,
        evaluator,
        metadata,
        signal_evaluation=signal_evaluation,
    )
    results = ensemble_evaluator.calculate(
        variables, compute_functions=True, compute_gradients=False
    )
    assert results
    assert isinstance(results[0], FunctionResults)
    _logger.info("Evaluation finished")
    send(
        EnOptEvent(
            event_type=EnOptEventType.FINISHED_ENSEMBLE_EVALUATOR,
            context=context,
            results=results,
        )
    )
