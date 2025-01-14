"""This module implements the default evaluator step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.config.validated_types import Array2D, ItemOrSet  # noqa: TC001
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.plan import Event
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults
from ropt.utils.scaling import scale_variables

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Plan


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

    The evaluator step uses the [`DefaultEvaluatorStepWith`]
    [ropt.plugins.plan.evaluator.DefaultEvaluatorStep.DefaultEvaluatorStepWith]
    configuration class to parse the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig] used to specify this step
    in a plan configuration.

    Note:
        The evaluator step is designed to serve as a base class for evaluator
        steps that may be initialized with different configuration objects. This
        can be done by providing a custom `parse_config` method to parse the
        `config` entry of the plan configuration into an
        [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. In addition, the
        `emit_event` method can be overridden to emit custom events, generally
        by adding additional information to its `data` field.
    """

    class DefaultEvaluatorStepWith(BaseModel):
        """Parameters used by the default evaluator step.

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

        Note: Variable Scaling
            Internally, the evaluator may use scaled variables as configured by
            the optimizer. When providing `values`, ensure they are unscaled;
            scaling is handled internally.

        Attributes:
            config: The optimizer configuration.
            tags:   Tags to add to the emitted events.
            values: Values to evaluate at.
        """

        config: str
        tags: ItemOrSet[str] = set()
        values: str | Array2D | None = None

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a default evaluator step.

        Args:
            config: The configuration of the step.
            plan:   The optimization plan that runs this step.
        """
        super().__init__(config, plan)

        self._with = self.DefaultEvaluatorStepWith.model_validate(config.with_)

    def run(self) -> None:
        """Run the evaluator step."""
        self._enopt_config = self.parse_config(self._with.config)

        self.emit_event(
            Event(event_type=EventType.START_EVALUATOR_STEP, tags=self._with.tags)
        )
        ensemble_evaluator = EnsembleEvaluator(
            self._enopt_config,
            self.plan.optimizer_context.evaluator,
            self.plan.plan_id,
            self.plan.result_id_iterator,
            self.plan.optimizer_context.plugin_manager,
        )

        variables = self._get_variables(self._enopt_config)
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

        self.emit_event(
            Event(
                event_type=EventType.FINISHED_EVALUATOR_STEP,
                tags=self._with.tags,
                data={"results": results, "exit_code": exit_code},
            )
        )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

    def parse_config(self, config: str) -> EnOptConfig:
        """Parse the configuration of the step.

        Returns:
            The parsed configuration.
        """
        config = self.plan.eval(config)
        if not isinstance(config, dict | EnOptConfig):
            msg = "No valid EnOpt configuration provided"
            raise TypeError(msg)
        return EnOptConfig.model_validate(config)

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
        """
        self.plan.emit_event(event)

    def _get_variables(self, config: EnOptConfig) -> NDArray[np.float64]:
        if self._with.values is not None:
            parsed_variables = self.plan.eval(self._with.values)
            if isinstance(parsed_variables, FunctionResults):
                return (
                    parsed_variables.evaluations.variables
                    if parsed_variables.evaluations.scaled_variables is None
                    else parsed_variables.evaluations.scaled_variables
                )
            if isinstance(parsed_variables, np.ndarray | list):
                parsed_variables = np.array(parsed_variables)
                scaled_variables = scale_variables(config, parsed_variables, axis=-1)
                return (
                    parsed_variables if scaled_variables is None else scaled_variables
                )
            if parsed_variables is not None:
                msg = f"`{self._with.values} does not contain variables."
                raise ValueError(msg)
        return self._enopt_config.variables.initial_values
