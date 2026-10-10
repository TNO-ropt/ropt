"""This module implements the default evaluator."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import numpy as np

from ropt._logging import get_logger
from ropt._scaling import scale
from ropt.core import EnsembleEvaluator
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.exceptions import OptimizerStop
from ropt.results import FunctionResults

from .base import ComputeStep

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ropt.components.concurrency import AbortSignal
    from ropt.components.evaluators import Evaluator
    from ropt.context import EnOptContext
    from ropt.results import Results


_logger = get_logger(__name__)


class EvaluationStep(ComputeStep[None]):
    """The default evaluation step compute step.

    Evaluates a batch of variable vectors (a single vector or a 2-D matrix
    where each row is a variable vector) and yields
    [`FunctionResults`][ropt.results.FunctionResults] objects. Emits
    `START_ENSEMBLE_EVALUATOR`, `START_EVALUATION`, `FINISHED_EVALUATION`,
    and `FINISHED_ENSEMBLE_EVALUATOR` events.

    See [Optimization Workflows](../advanced/workflows.md#events-emitted-by-evaluationstep)
    for the full event lifecycle description.
    """

    def __init__(
        self, *, evaluator: Evaluator, abort_signal: AbortSignal | None = None
    ) -> None:
        """Initialize a default evaluator.

        Args:
            evaluator:   The evaluator object to run function evaluations.
            abort_signal: An optional signal that cuts this step off.
        """
        super().__init__(abort_signal=abort_signal)
        self._evaluator = evaluator

    @override
    def _run(
        self,
        context: EnOptContext,
        variables: ArrayLike,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Run the ensemble evaluation.

        `metadata` is attached to the emitted
        [`FunctionResults`][ropt.results.FunctionResults] via the
        `FINISHED_EVALUATION` event.

        Args:
            context:   Optimizer context.
            variables: Variable vector(s) to evaluate.
            metadata:  Optional dictionary attached to emitted results.

        Raises:
            ValueError: If the input variables have the wrong shape.
        """
        # A single vector is accepted as a batch of one, so everything below
        # works on a matrix.
        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=2)
        if variables.shape[-1] != context.variables.variable_count:
            msg = "The input variables have the wrong shape"
            raise ValueError(msg)
        # Claim the context: it is single use, so two runs cannot share one.
        context.lock()

        _logger.info("Starting evaluation")
        try:
            results = self._evaluate(context, variables, metadata)
        except OptimizerStop:
            # The batch was aborted on this step's signal. There is nothing to
            # report, and the caller reads the signal for why.
            _logger.info("Evaluation aborted")
            return
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

        variables = scale(
            variables, context.variables.scales, context.variables.offsets
        )

        self._context = context
        self._metadata = metadata
        ensemble_evaluator = EnsembleEvaluator(
            context,
            self._evaluator.eval,
            metadata,
            signal_evaluation=self._signal_evaluation,
        )

        results = ensemble_evaluator.calculate(
            variables, compute_functions=True, compute_gradients=False
        )

        assert results
        assert isinstance(results[0], FunctionResults)

        return results

    def _signal_evaluation(self, results: tuple[Results, ...] | None = None) -> None:
        # Called by the ensemble evaluator around every evaluation: without
        # results before one starts, with them once it has finished.
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
                )
            )
