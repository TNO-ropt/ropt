"""This module implements the default evaluator step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.plan import Event
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.config.enopt import EnOptConfig
    from ropt.plan import Plan


class DefaultEnsembleEvaluatorStep(PlanStep):
    """The default ensemble evaluator step for optimization plans.

    This step performs one or more ensemble evaluations based on the provided
    `variables`. It yields a tuple of
    [`FunctionResults`][ropt.results.FunctionResults] objects, one for each
    input variable vector evaluated.

    The step emits the following events:

    - [`START_ENSEMBLE_EVALUATOR_STEP`][ropt.enums.EventType.START_ENSEMBLE_EVALUATOR_STEP]:
      Emitted before the evaluation process begins.
    - [`START_EVALUATION`][ropt.enums.EventType.START_EVALUATION]: Emitted
      just before the underlying ensemble evaluation is called.
    - [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION]: Emitted
      after the evaluation completes, carrying the generated `FunctionResults`
      in its `data` dictionary under the key `"results"`. Event handlers
      typically listen for this event.
    - [`FINISHED_ENSEMBLE_EVALUATOR_STEP`][ropt.enums.EventType.FINISHED_ENSEMBLE_EVALUATOR_STEP]:
      Emitted after the entire step, including result emission, is finished.
    """

    def __init__(
        self,
        plan: Plan,
        tags: set[str] | None = None,
    ) -> None:
        """Initialize a default evaluator step.

        Args:
            plan: The plan that runs this step.
            tags: Optional tags
        """
        super().__init__(plan, tags)

    def run_step_from_plan(
        self,
        config: EnOptConfig,
        variables: ArrayLike | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizerExitCode:
        """Run the ensemble evaluator step to perform ensemble evaluations.

        This method executes the core logic of the ensemble evaluator step. It
        requires an optimizer configuration
        ([`EnOptConfig`][ropt.config.enopt.EnOptConfig]) and optionally accepts
        specific variable vectors to evaluate.

        If `variables` are not provided, the initial values specified in the
        `config` are used. If `variables` are provided as a 2D array, each row
        is treated as a separate vector for evaluation.

        If `metadata` is provided, it is attached to the
        [`Results`][ropt.results.Results] objects emitted via the
        `FINISHED_EVALUATION` event.

        Args:
            config:    Optimizer configuration.
            variables: Optional variable vector(s) to evaluate.
            metadata:  Optional dictionary to attach to emitted `FunctionResults`.

        Returns:
            An [`OptimizerExitCode`][ropt.enums.OptimizerExitCode] indicating the outcome.
        """
        self._event_handlers = self.plan.get_event_handlers(
            self,
            {
                EventType.START_ENSEMBLE_EVALUATOR_STEP,
                EventType.FINISHED_ENSEMBLE_EVALUATOR_STEP,
                EventType.START_EVALUATION,
                EventType.FINISHED_EVALUATION,
            },
        )

        event_data: dict[str, Any] = {"config": config}

        self._emit_event(
            Event(event_type=EventType.START_ENSEMBLE_EVALUATOR_STEP, data=event_data)
        )

        if variables is None:
            variables = config.variables.initial_values
        else:
            variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)
            if (
                config.transforms is not None
                and config.transforms.variables is not None
            ):
                variables = config.transforms.variables.to_optimizer(variables)

        evaluator = self.plan.get_evaluator(self)
        ensemble_evaluator = EnsembleEvaluator(
            config, evaluator.eval, self.plan.plugin_manager
        )

        exit_code = OptimizerExitCode.EVALUATOR_STEP_FINISHED

        self._emit_event(Event(event_type=EventType.START_EVALUATION, data=event_data))
        try:
            results = ensemble_evaluator.calculate(
                variables, compute_functions=True, compute_gradients=False
            )
        except OptimizationAborted as exc:
            exit_code = exc.exit_code

        assert results
        assert isinstance(results[0], FunctionResults)
        if results[0].functions is None:
            exit_code = OptimizerExitCode.TOO_FEW_REALIZATIONS

        if metadata is not None:
            for item in results:
                item.metadata = deepcopy(metadata)

        event_data["results"] = results

        self._emit_event(
            Event(event_type=EventType.FINISHED_EVALUATION, data=event_data)
        )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

        self._emit_event(
            Event(
                event_type=EventType.FINISHED_ENSEMBLE_EVALUATOR_STEP, data=event_data
            )
        )

        return exit_code

    def _emit_event(self, event: Event) -> None:
        for handler in self._event_handlers.get(event.event_type, []):
            handler(event)
