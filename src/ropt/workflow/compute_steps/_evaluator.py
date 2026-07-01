"""This module implements the default evaluator."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.core import EnsembleEvaluator
from ropt.enums import EnOptEventType, ExitCode
from ropt.events import EnOptEvent
from ropt.exceptions import Abort
from ropt.exit_info import ExitInfo
from ropt.results import FunctionResults

from .base import ComputeStep

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.context import EnOptContext
    from ropt.workflow.evaluators import Evaluator


class EvaluationStep(ComputeStep):
    """The default evaluation step compute step.

    Evaluates a batch of variable vectors (a single vector or a 2-D matrix
    where each row is a variable vector) and yields
    [`FunctionResults`][ropt.results.FunctionResults] objects. Emits
    `START_ENSEMBLE_EVALUATOR`, `START_EVALUATION`, `FINISHED_EVALUATION`,
    and `FINISHED_ENSEMBLE_EVALUATOR` events.

    See [Optimization Workflows](../usage/workflows.md#events-emitted-by-evualuationstep)
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
    ) -> ExitInfo:
        """Run the ensemble evaluation.

        Args:
            context:    Optimizer context.
            variables:  Variable vector(s) to evaluate.
            metadata:   Optional dictionary attached to emitted
                        [`FunctionResults`][ropt.results.FunctionResults] via the
                       `FINISHED_EVALUATION` event.

        Returns:
            An [`ExitInfo`][ropt.exit_info.ExitInfo] describing the outcome.

        Raises:
            ValueError: If the input variables have the wrong shape.
        """
        context.lock()

        self._emit_event(
            EnOptEvent(
                event_type=EnOptEventType.START_ENSEMBLE_EVALUATOR, context=context
            )
        )

        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=2)
        if variables.shape[-1] != context.variables.variable_count:
            msg = "The input variables have the wrong shape"
            raise ValueError(msg)
        for transform in context.variable_transforms:
            variables = transform.to_optimizer(variables)

        ensemble_evaluator = EnsembleEvaluator(context, self._evaluator.eval)

        exit_info = ExitInfo(exit_code=ExitCode.ENSEMBLE_EVALUATOR_FINISHED)

        self._emit_event(
            EnOptEvent(event_type=EnOptEventType.START_EVALUATION, context=context)
        )
        try:
            results = ensemble_evaluator.calculate(
                variables, compute_functions=True, compute_gradients=False
            )
        except Abort as exc:
            exit_info = exc.info

        assert results
        assert isinstance(results[0], FunctionResults)
        if results[0].functions is None:
            exit_info = ExitInfo(exit_code=ExitCode.TOO_FEW_REALIZATIONS)

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

        self._emit_event(
            EnOptEvent(
                event_type=EnOptEventType.FINISHED_ENSEMBLE_EVALUATOR,
                context=context,
                results=results,
            )
        )

        return exit_info

    def _emit_event(self, event: EnOptEvent) -> None:
        for handler in self.event_handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)
