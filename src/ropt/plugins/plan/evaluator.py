"""This module implements the default evaluator step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.config.enopt import EnOptConfig
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.plan import Event
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.plan import Plan
    from ropt.transforms import OptModelTransforms


class DefaultEvaluatorStep(PlanStep):
    """The default evaluator step.

    This step performs a single ensemble evaluation, yielding one or more
    [`FunctionResults`][ropt.results.FunctionResults] objects. The evaluation
    can process multiple variable vectors, each of which is evaluated
    separately, producing an individual results object for each vector.

    Before executing the evaluator step, a
    [`START_EVALUATOR_STEP`][ropt.enums.EventType.START_EVALUATOR_STEP] event is
    emitted. After the evaluator step finishes, an
    [`FINISHED_EVALUATOR_STEP`][ropt.enums.EventType.FINISHED_EVALUATOR_STEP]
    event is emitted. Result handlers should respond to the latter event to
    process the generated results.
    """

    def __init__(
        self,
        plan: Plan,
        *,
        tag: str | None = None,
    ) -> None:
        """Initialize a default evaluator step.

        The `tag` field allows an optional label to be attached to each result,
        which can assist result handlers in filtering relevant results.

        Args:
            plan: The plan that runs this step.
            tag:  Tag to add to the emitted events.
        """
        super().__init__(plan)
        self._tag = tag

    def run(  # type: ignore[override]
        self,
        config: Any,  # noqa: ANN401
        transforms: OptModelTransforms | None = None,
        variables: ArrayLike | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizerExitCode:
        """Run the evaluator step.

        The
        [`DefaultEvaluatorStep`][ropt.plugins.plan.evaluator.DefaultEvaluatorStep]
        requires an optimizer configuration; the `variables` parameter is
        optional. The configuration  object must be an
        [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object, or a dictionary
        that can be parsed into such an object. If no `variables` are provided,
        the initial values specified by the optimizer configuration are used. If
        `values` is given, it may be a single vector or a two-dimensional array.
        In the latter case, each row of the matrix is treated as a separate set
        of values to be evaluated.

        Args:
            config:     The optimizer configuration.
            transforms: Optional transforms object.
            variables: Variables to evaluate.
            metadata:   Optional metadata to add to events.
        """
        config = EnOptConfig.model_validate(config, context=transforms)

        self.emit_event(
            Event(
                event_type=EventType.START_EVALUATOR_STEP,
                config=config,
                tag=self._tag,
            )
        )

        if variables is None:
            variables = config.variables.initial_values
        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)

        ensemble_evaluator = EnsembleEvaluator(
            config,
            transforms,
            self.plan.optimizer_context.evaluator,
            self.plan.optimizer_context.plugin_manager,
        )

        exit_code = OptimizerExitCode.EVALUATION_STEP_FINISHED

        self.emit_event(
            Event(
                event_type=EventType.START_EVALUATION,
                config=config,
                tag=self._tag,
            )
        )
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

        data: dict[str, Any] = {}
        if transforms is not None:
            data["transformed_results"] = results
            data["results"] = [
                item.transform_from_optimizer(transforms) for item in results
            ]
        else:
            data["results"] = results

        self.emit_event(
            Event(
                event_type=EventType.FINISHED_EVALUATION,
                config=config,
                tag=self._tag,
                data=data,
            )
        )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

        self.emit_event(
            Event(
                event_type=EventType.FINISHED_EVALUATOR_STEP,
                config=config,
                tag=self._tag,
            )
        )

        return exit_code

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
        """
        self.plan.emit_event(event)
