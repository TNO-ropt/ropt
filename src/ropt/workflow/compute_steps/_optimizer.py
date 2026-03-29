"""This module implements the default optimizer compute step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.core import EnsembleEvaluator as CoreEnsembleEvaluator
from ropt.core import EnsembleOptimizer as CoreEnsembleOptimizer
from ropt.enums import EnOptEventType, ExitCode
from ropt.events import EnOptEvent

from .base import ComputeStep

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.context import EnOptContext
    from ropt.results import Results
    from ropt.workflow.evaluators import Evaluator


MetaDataType = dict[str, int | float | bool | str]


class EnsembleOptimizer(ComputeStep):
    """The default optimizer compute step.

    This compute step executes an optimization algorithm based on a provided
    configuration ([`EnOptContext`][ropt.context.EnOptContext] or a compatible
    dictionary). It iteratively performs function and potentially gradient
    evaluations, yielding a sequence of
    [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults] objects.

    While initial variable values are typically specified in the configuration,
    they can be overridden by passing them directly to the `run` method.

    The following events are emitted during execution:

    - [`START_OPTIMIZER`][ropt.enums.EnOptEventType.START_OPTIMIZER]:
      Emitted just before the optimization process begins.
    - [`START_EVALUATION`][ropt.enums.EnOptEventType.START_EVALUATION]: Emitted
      immediately before an ensemble evaluation (for functions or gradients)
      is requested from the underlying optimizer.
    - [`FINISHED_EVALUATION`][ropt.enums.EnOptEventType.FINISHED_EVALUATION]: Emitted
      after an evaluation completes. This event carries the generated
      [`Results`][ropt.results.Results] object(s) in its `data` dictionary
      under the key `"results"`. Event handlers typically listen for this event
      to process or track optimization progress.
    - [`FINISHED_OPTIMIZER`][ropt.enums.EnOptEventType.FINISHED_OPTIMIZER]:
      Emitted after the entire optimization process concludes (successfully,
      or due to termination conditions or errors).
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
    ) -> ExitCode:
        """Run the compute step to perform an optimization.

        This method executes the core logic of the optimizer compute step. It
        requires an optimizer context
        ([`EnOptContext`][ropt.context.EnOptContext]) and optionally accepts
        specific initial variable vectors and metadata.

        If `metadata` is provided, it is attached to the
        [`Results`][ropt.results.Results] objects emitted via the
        `FINISHED_EVALUATION` event.

        Args:
            context:    The optimizer context.
            variables:  Optional initial variable vector(s) to start from.
            metadata:   Optional dictionary to attach to emitted `Results`.

        Returns:
            An exit code indicating the outcome of the optimization.

        Raises:
            ValueError:   If the input variables have the wrong shape.
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

        ensemble_evaluator = CoreEnsembleEvaluator(
            self._context,
            self._evaluator.eval,
        )
        ensemble_optimizer = CoreEnsembleOptimizer(
            context=self._context,
            ensemble_evaluator=ensemble_evaluator,
            signal_evaluation=self._signal_evaluation,
        )
        exit_code = ensemble_optimizer.start(variables)

        self._emit_event(
            EnOptEvent(event_type=EnOptEventType.FINISHED_OPTIMIZER, context=context)
        )

        return exit_code

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
