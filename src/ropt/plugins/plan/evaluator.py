"""This module implements the default evaluator step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.config.enopt import EnOptConfig
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.plan import Event
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults

from ._utils import _get_set

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

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
        config: Any,  # noqa: ANN401
        transforms: OptModelTransforms | None = None,
        tags: str | set[str] | None = None,
    ) -> None:
        """Initialize a default evaluator step.

        The
        [`DefaultEvaluatorStep`][ropt.plugins.plan.evaluator.DefaultEvaluatorStep]
        requires an optimizer configuration; the `tags` and `values` parameters
        are optional. The configuration  object must be an
        [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object, or a dictionary
        that can be parsed into such an object. If no `values` are provided, the
        initial values specified by the optimizer configuration are used. If
        `values` is given, it may be a single vector or a two-dimensional array.
        In the latter case, each row of the matrix is treated as a separate set
        of values to be evaluated.

        The `tags` field allows optional labels to be attached to each result,
        which can assist result handlers in filtering relevant results.

        Args:
            plan:       The plan that runs this step.
            config:     The optimizer configuration.
            transforms: Optional transforms object.
            tags:       Tags to add to the emitted events.
        """
        super().__init__(plan)
        self._config = EnOptConfig.model_validate(config)
        self._transforms = transforms
        self._tags = _get_set(tags)

    def run(  # type: ignore[override]
        self,
        *,
        variables: ArrayLike | None = None,
    ) -> OptimizerExitCode:
        """Run the evaluator step."""
        variables = (
            self._config.variables.initial_values
            if variables is None
            else np.array(np.asarray(variables, dtype=np.float64), ndmin=1)
        )
        return self._run(self._config, self._transforms, variables)

    def _run(
        self,
        enopt_config: EnOptConfig,
        transforms: OptModelTransforms | None,
        variables: NDArray[np.float64],
    ) -> OptimizerExitCode:
        for event_type in (EventType.START_EVALUATOR_STEP, EventType.START_EVALUATION):
            self.emit_event(
                Event(
                    event_type=event_type,
                    config=enopt_config,
                    tags=self._tags,
                )
            )

        ensemble_evaluator = EnsembleEvaluator(
            enopt_config,
            transforms,
            self.plan.optimizer_context.evaluator,
            self.plan.plan_id,
            self.plan.optimizer_context.eval_id_iter,
            self.plan.optimizer_context.plugin_manager,
        )

        exit_code = OptimizerExitCode.EVALUATION_STEP_FINISHED

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

        data: dict[str, Any] = {}
        if transforms is not None:
            data["transformed_results"] = results
            data["results"] = [
                item.transform_from_optimizer(transforms) for item in results
            ]
        else:
            data["results"] = results
        for event_type in (
            EventType.FINISHED_EVALUATION,
            EventType.FINISHED_EVALUATOR_STEP,
        ):
            self.emit_event(
                Event(
                    event_type=event_type,
                    config=enopt_config,
                    tags=self._tags,
                    data=data,
                )
            )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

        return exit_code

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
        """
        self.plan.emit_event(event)
