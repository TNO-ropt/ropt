"""This module implements the default evaluator step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

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

        The `data` field can be used to pass additional information via the `data`
        field of the events emitted by the optimizer step. Avoid the use of `results`
        and `exit_code` in this field, as these are already passed in the event data.

        Attributes:
            config: The optimizer configuration.
            tags:   Tags to add to the emitted events.
            values: Values to evaluate at.
            data:   Data to pass via events.
        """

        config: str
        tags: ItemOrSet[str] = set()
        values: str | Array2D | None = None
        data: dict[str, Any] = {}

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
        variables = self._get_variables()
        config = self.plan.eval(self._with.config)
        if not isinstance(config, dict | EnOptConfig):
            msg = "No valid EnOpt configuration provided"
            raise TypeError(msg)
        enopt_config = EnOptConfig.model_validate(config)
        if variables is None:
            variables = enopt_config.variables.initial_values
        self._run(enopt_config, variables)

    def _run(self, enopt_config: EnOptConfig, variables: NDArray[np.float64]) -> None:
        for event_type in (EventType.START_EVALUATOR_STEP, EventType.START_EVALUATION):
            self.emit_event(
                Event(
                    event_type=event_type,
                    config=enopt_config,
                    tags=self._with.tags,
                    data=deepcopy(self._with.data),
                )
            )

        ensemble_evaluator = EnsembleEvaluator(
            enopt_config,
            self.plan.optimizer_context.evaluator,
            self.plan.plan_id,
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

        data = deepcopy(self._with.data)
        data["exit_code"] = exit_code
        if enopt_config.transforms is not None:
            data["transformed_results"] = results
            data["results"] = [
                item.transform_back(enopt_config.transforms) for item in results
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
                    tags=self._with.tags,
                    data=data,
                )
            )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
        """
        self.plan.emit_event(event)

    def _get_variables(self) -> NDArray[np.float64] | None:
        if self._with.values is not None:
            parsed_variables = self.plan.eval(self._with.values)
            match parsed_variables:
                case FunctionResults():
                    return parsed_variables.evaluations.variables
                case np.ndarray() | list():
                    return np.asarray(parsed_variables, dtype=np.float64)
                case None:
                    return None
                case _:
                    msg = f"`{self._with.values}` does not contain variables."
                    raise ValueError(msg)
        return None
