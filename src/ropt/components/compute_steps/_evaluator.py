"""This module implements the default evaluator."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._logging import get_logger
from ropt.core import EnsembleEvaluator
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import FunctionResults

from .base import ComputeStep

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ropt.components.evaluators import Evaluator
    from ropt.context import EnOptContext
    from ropt.results import Results


_logger = get_logger(__name__)


class EvaluationStep(ComputeStep):
    """The default evaluation step compute step.

    Evaluates a batch of variable vectors (a single vector or a 2-D matrix
    where each row is a variable vector) and yields
    [`FunctionResults`][ropt.results.FunctionResults] objects. Emits
    `START_ENSEMBLE_EVALUATOR`, `START_EVALUATION`, `FINISHED_EVALUATION`,
    and `FINISHED_ENSEMBLE_EVALUATOR` events.

    See [Optimization Workflows](../low_level/workflows.md#events-emitted-by-evualuationstep)
    for the full event lifecycle description.
    """

    def __init__(self, *, evaluator: Evaluator) -> None:
        """Initialize a default evaluator.

        Args:
            evaluator: The evaluator object to run function evaluations.
        """
        super().__init__()
        self._evaluator = evaluator

    def run(
        self,
        context: EnOptContext,
        variables: ArrayLike,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Run the ensemble evaluation.

        Args:
            context:   Optimizer context.
            variables: Variable vector(s) to evaluate.
            metadata:  Optional dictionary attached to emitted
                       [`FunctionResults`][ropt.results.FunctionResults] via the
                       `FINISHED_EVALUATION` event.

        Raises:
            ValueError: If the input variables have the wrong shape.
        """
        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=2)
        if variables.shape[-1] != context.variables.variable_count:
            msg = "The input variables have the wrong shape"
            raise ValueError(msg)
        context.lock()

        _logger.info("Starting evaluation")
        results = self._evaluate(context, variables, metadata)
        _logger.info("Evaluation finished")
        self._emit_event(
            EnOptEvent(
                event_type=EnOptEventType.FINISHED_ENSEMBLE_EVALUATOR,
                context=context,
                results=results,
            )
        )

    def _evaluate(
        self,
        context: EnOptContext,
        variables: NDArray[np.float64],
        metadata: dict[str, Any] | None,
    ) -> tuple[Results, ...]:
        self._emit_event(
            EnOptEvent(
                event_type=EnOptEventType.START_ENSEMBLE_EVALUATOR, context=context
            )
        )

        for transform in context.variable_transforms:
            variables = transform.to_optimizer(variables)

        ensemble_evaluator = EnsembleEvaluator(context, self._evaluator.eval)

        self._emit_event(
            EnOptEvent(event_type=EnOptEventType.START_EVALUATION, context=context)
        )
        results = ensemble_evaluator.calculate(
            variables, compute_functions=True, compute_gradients=False
        )

        assert results
        assert isinstance(results[0], FunctionResults)

        if metadata is not None:
            for item in results:
                item.metadata = deepcopy(metadata)

        self._emit_event(
            EnOptEvent(
                event_type=EnOptEventType.FINISHED_EVALUATION,
                context=context,
                results=results,
            )
        )

        return results

    def _emit_event(self, event: EnOptEvent) -> None:
        for handler in self.event_handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)
