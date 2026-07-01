"""This module implements the default optimizer compute step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.core import EnsembleEvaluator
from ropt.core import EnsembleOptimizer as CoreEnsembleOptimizer
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent

from .base import ComputeStep

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.context import EnOptContext
    from ropt.exit_info import ExitInfo
    from ropt.results import Results
    from ropt.workflow.evaluators import Evaluator


MetaDataType = dict[str, int | float | bool | str]


class OptimizationStep(ComputeStep):
    """The default optimizer compute step.

    Executes an optimization algorithm, iteratively performing function and
    gradient evaluations. Emits `START_OPTIMIZER`, `START_EVALUATION`,
    `FINISHED_EVALUATION`, and `FINISHED_OPTIMIZER` events.

    See [Optimization Workflows](../usage/workflows.md#events-emitted-by-optimizationstep)
    for the full event lifecycle description.
    """

    def __init__(self, *, evaluator: Evaluator) -> None:
        """Initialize a default optimizer.

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
        """Run the optimization.

        Args:
            context:    The optimizer context.
            variables:  Initial variable vector(s).
            metadata:   Optional dictionary attached to emitted
                [`Results`][ropt.results.Results] via the `FINISHED_EVALUATION`
                event.

        Returns:
            An exit info object describing the outcome of the optimization.

        Raises:
            ValueError: If the input variables have the wrong shape.
        """
        context.lock()

        self._context = context
        self._metadata = metadata

        self._emit_event(
            EnOptEvent(event_type=EnOptEventType.START_OPTIMIZER, context=context)
        )

        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)
        if variables.shape != (self._context.variables.variable_count,):
            msg = "The input variables have the wrong shape"
            raise ValueError(msg)
        for transform in context.variable_transforms:
            variables = transform.to_optimizer(variables)

        ensemble_evaluator = EnsembleEvaluator(
            self._context,
            self._evaluator.eval,
        )
        ensemble_optimizer = CoreEnsembleOptimizer(
            context=self._context,
            ensemble_evaluator=ensemble_evaluator,
            signal_evaluation=self._signal_evaluation,
        )
        exit_info = ensemble_optimizer.start(variables)

        self._emit_event(
            EnOptEvent(event_type=EnOptEventType.FINISHED_OPTIMIZER, context=context)
        )

        return exit_info

    def _emit_event(self, event: EnOptEvent) -> None:
        for handler in self.event_handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)

    def _signal_evaluation(self, results: tuple[Results, ...] | None = None) -> None:
        if results is None:
            self._emit_event(
                EnOptEvent(
                    event_type=EnOptEventType.START_EVALUATION, context=self._context
                )
            )
        else:
            if self._metadata is not None:
                for item in results:
                    item.metadata = deepcopy(self._metadata)

            self._emit_event(
                EnOptEvent(
                    event_type=EnOptEventType.FINISHED_EVALUATION,
                    context=self._context,
                    results=results,
                ),
            )
