"""This module implements the default evaluator step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, ExitCode
from ropt.exceptions import StepAborted
from ropt.plan import Event
from ropt.plugins.plan.base import Evaluator, PlanStep
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.config import EnOptConfig
    from ropt.plugins import PluginManager
    from ropt.transforms import OptModelTransforms


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

    def __init__(self, *, evaluator: Evaluator, plugin_manager: PluginManager) -> None:
        """Initialize a default evaluator step.

        Args:
            evaluator:      The evaluator object to run function evaluations.
            plugin_manager: The plugin manager used to retrieve optimizer components
        """
        super().__init__()
        self._evaluator = evaluator
        self._plugin_manager = plugin_manager

    def run(
        self,
        config: EnOptConfig,
        variables: ArrayLike,
        *,
        transforms: OptModelTransforms | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExitCode:
        """Run the ensemble evaluator step to perform ensemble evaluations.

        This method executes the core logic of the ensemble evaluator step. It
        requires an optimizer configuration
        ([`EnOptConfig`][ropt.config.EnOptConfig]) and optionally accepts
        specific variable vectors to evaluate.

        If `metadata` is provided, it is attached to the
        [`Results`][ropt.results.Results] objects emitted via the
        `FINISHED_EVALUATION` event.

        Args:
            config:     Optimizer configuration.
            variables:  Variable vector(s) to evaluate.
            transforms: Optional transforms to apply to the variables,
                        objectives, and constraints.
            metadata:   Optional dictionary to attach to emitted `FunctionResults`.

        Returns:
            An [`ExitCode`][ropt.enums.ExitCode] indicating the outcome.
        """
        event_data: dict[str, Any] = {"config": config, "transforms": transforms}

        self._emit_event(
            Event(event_type=EventType.START_ENSEMBLE_EVALUATOR_STEP, data=event_data)
        )

        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=2)
        if variables.shape[-1] != config.variables.variable_count:
            msg = "The input variables have the wrong shape"
            raise ValueError(msg)
        if transforms is not None and transforms.variables is not None:
            variables = transforms.variables.to_optimizer(variables)

        ensemble_evaluator = EnsembleEvaluator(
            config, transforms, self._evaluator.eval, self._plugin_manager
        )

        exit_code = ExitCode.EVALUATOR_STEP_FINISHED

        self._emit_event(Event(event_type=EventType.START_EVALUATION, data=event_data))
        try:
            results = ensemble_evaluator.calculate(
                variables, compute_functions=True, compute_gradients=False
            )
        except StepAborted as exc:
            exit_code = exc.exit_code

        assert results
        assert isinstance(results[0], FunctionResults)
        if results[0].functions is None:
            exit_code = ExitCode.TOO_FEW_REALIZATIONS

        if metadata is not None:
            for item in results:
                item.metadata = deepcopy(metadata)

        event_data["results"] = results

        self._emit_event(
            Event(event_type=EventType.FINISHED_EVALUATION, data=event_data)
        )

        self._emit_event(
            Event(
                event_type=EventType.FINISHED_ENSEMBLE_EVALUATOR_STEP, data=event_data
            )
        )

        return exit_code

    def _emit_event(self, event: Event) -> None:
        for handler in self.event_handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)
