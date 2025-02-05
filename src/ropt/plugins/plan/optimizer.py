"""This module implements the default optimizer step."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.config.plan import PlanConfig  # noqa: TC001
from ropt.config.validated_types import Array1D, ItemOrSet, ItemOrTuple  # noqa: TC001
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.optimization import EnsembleOptimizer
from ropt.plan import Event, Plan
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.plan import PlanStepConfig
    from ropt.results import Results
    from ropt.transforms import OptModelTransforms

MetaDataType = dict[str, int | float | bool | str]


class DefaultOptimizerStep(PlanStep):
    """The default optimizer step.

    The optimizer step performs an optimization, yielding a sequence of
    [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults] objects. The optimizer is
    configured using an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object or
    a dictionary that can be parsed into such an object. While the initial values
    for optimization are typically specified in the configuration, they can be
    overridden by providing them directly.

    The optimizer step emits several signals:

    - [`START_OPTIMIZER_STEP`][ropt.enums.EventType.START_OPTIMIZER_STEP]:
      Emitted before the optimization starts.
    - [`FINISHED_OPTIMIZER_STEP`][ropt.enums.EventType.FINISHED_OPTIMIZER_STEP]:
      Emitted after the optimization finishes.
    - [`START_EVALUATION`][ropt.enums.EventType.START_EVALUATION]: Emitted
      before a function or gradient evaluation.
    - [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION]: Emitted
      after a function or gradient evaluation.

      The `FINISHED_EVALUATION` signal is particularly important as it passes
      the generated [`Results`][ropt.results.Results] objects. Result handlers
      specified in the plan will respond to this signal to process those results.

    The optimizer step supports nested optimizations, where each function
    evaluation in the optimization triggers a nested optimization plan that
    should produce the result for the function evaluation.

    The optimizer step utilizes the
    [`DefaultOptimizerStepWith`]
    [ropt.plugins.plan.optimizer.DefaultOptimizerStep.DefaultOptimizerStepWith]
    configuration class to parse the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig] used to specify this step
    in a plan configuration.
    """

    class NestedPlanConfig(BaseModel):
        """Parameters needed by a nested plan.

        A nested plan will be executed by its parent plan each time it evaluates
        a function. While a nested plan is a standard [`Plan`][ropt.plan.Plan],
        it must adhere to certain rules:

        - It must accept at least one input variable that holds the variable
          values at which the parent plan intends to evaluate a function.
        - It must return one result, which the parent plan will accept as the
          result of the function evaluation.

        Extra inputs to the nested plan can be specified using the
        `extra_inputs` field. The nested plan must account for these extra
        inputs accordingly.

        Nested plans run independently, similar to a standard plan. They
        typically produce [`Results`][ropt.results.Results], which may be
        processed using result handlers defined in the nested plan. Once these
        handlers have executed, the results are also 'bubbled' up to the parent
        plan, where they are processed with its own handlers.

        Attributes:
            plan:         The nested plan.
            extra_inputs: Extra inputs passed to the plan.
        """

        plan: PlanConfig
        extra_inputs: ItemOrTuple[Any] = ()

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    class DefaultOptimizerStepWith(BaseModel):
        """Parameters used by the default optimizer step.

        The [`DefaultOptimizerStep`][ropt.plugins.plan.optimizer.DefaultOptimizerStep]
        requires an optimizer configuration; all other parameters are optional.
        The configuration object must be an
        [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object or a dictionary
        that can be parsed into such an object. Initial values can be provided
        optionally; if not specified, the initial values defined by the
        optimizer configuration will be used.

        The `exit_code_var` field can be used to specify the name of a plan
        variable where the [`exit code`][ropt.enums.OptimizerExitCode] is
        stored, which the optimizer returns upon completion.

        The `tags` field allows optional labels to be attached to each result,
        assisting result handlers in filtering relevant results.

        The `data` field can be used to pass additional information via the `data`
        field of the events emitted by the optimizer step. Avoid the use of `results`
        and `exit_code` in this field, as these are already passed in the event data.

        The `nested_optimization_plan` is parsed as a [`NestedPlanConfig`]
        [ropt.plugins.plan.optimizer.DefaultOptimizerStep.NestedPlanConfig] to
        define an optional nested optimization procedure.

        Attributes:
            config:                The optimizer configuration.
            tags:                  Tags to add to the emitted events.
            initial_values:        The initial values for the optimizer.
            exit_code_var:         Name of the variable to store the exit code.
            data:                  Data to pass via events.
            nested_optimization:   Optional nested optimization plan configuration.
        """

        config: str
        tags: ItemOrSet[str] = set()
        initial_values: str | Array1D | None = None
        exit_code_var: str | None = None
        nested_optimization: DefaultOptimizerStep.NestedPlanConfig | None = None
        data: dict[str, Any] = {}

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a default optimizer step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)

        self._with = self.DefaultOptimizerStepWith.model_validate(config.with_)

    def run(self) -> None:
        """Run the optimizer step.

        Returns:
            Whether a user abort occurred.
        """
        variables = self._get_initial_variables()
        config = self.plan.eval(self._with.config)
        if not isinstance(config, dict | EnOptConfig):
            msg = "No valid EnOpt configuration provided"
            raise TypeError(msg)
        enopt_config = EnOptConfig.model_validate(config)
        if variables is None:
            variables = enopt_config.variables.initial_values
        self._run(enopt_config, variables)

    def _run(self, enopt_config: EnOptConfig, variables: NDArray[np.float64]) -> None:
        self.emit_event(
            Event(
                event_type=EventType.START_OPTIMIZER_STEP,
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

        ensemble_optimizer = EnsembleOptimizer(
            enopt_config=enopt_config,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self.plan.optimizer_context.plugin_manager,
            nested_optimizer=(
                self._run_nested_plan
                if self._with.nested_optimization is not None
                else None
            ),
            signal_evaluation=partial(
                self._signal_evaluation, transforms=enopt_config.transforms
            ),
        )

        if (
            ensemble_optimizer.is_parallel
            and self._with.nested_optimization is not None
        ):
            msg = "Nested optimization detected: parallel evaluation not supported. "
            raise RuntimeError(msg)

        exit_code = ensemble_optimizer.start(variables)

        if self._with.exit_code_var is not None:
            self.plan[self._with.exit_code_var] = exit_code

        data = deepcopy(self._with.data)
        data["exit_code"] = exit_code
        self.emit_event(
            Event(
                event_type=EventType.FINISHED_OPTIMIZER_STEP,
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

    def _signal_evaluation(
        self,
        results: tuple[Results, ...] | None = None,
        *,
        exit_code: OptimizerExitCode | None = None,
        transforms: OptModelTransforms | None = None,
    ) -> None:
        """Called before and after the optimizer finishes an evaluation.

        Before the evaluation starts, this method is called with the `results`
        argument set to `None`. When an evaluation is has finished, it is called
        with `results` set to the results of the evaluation.

        Args:
            results:    The results produced by the evaluation.
            exit_code:  An exit code if that may be set if the evaluation completed.
            transforms: Optional transforms for the results.
        """
        if results is None:
            self.emit_event(
                Event(
                    event_type=EventType.START_EVALUATION,
                    tags=self._with.tags,
                    data=deepcopy(self._with.data),
                )
            )
        else:
            data = deepcopy(self._with.data)
            data["exit_code"] = exit_code
            if transforms is not None:
                data["scaled_results"] = results
                data["results"] = [item.transform_back(transforms) for item in results]
            else:
                data["results"] = results
            self.emit_event(
                Event(
                    event_type=EventType.FINISHED_EVALUATION,
                    tags=self._with.tags,
                    data=data,
                ),
            )

    def _run_nested_plan(
        self, variables: NDArray[np.float64]
    ) -> tuple[FunctionResults | None, bool]:
        """Run a  nested plan.

        Args:
            variables: variables to set in the nested plan.

        Returns:
            The variables generated by the nested plan.
        """
        if self._with.nested_optimization is None:
            return None, False
        plan = self.plan.spawn(self._with.nested_optimization.plan)
        extra_inputs = [
            self._plan.eval(item)
            for item in self._with.nested_optimization.extra_inputs
        ]
        results = plan.run(variables, *extra_inputs)
        assert len(results) == 1
        assert results[0] is None or isinstance(results[0], FunctionResults)
        if plan.aborted:
            self.plan.abort()
        return results[0], plan.aborted

    def _get_initial_variables(self) -> NDArray[np.float64] | None:
        if self._with.initial_values is not None:
            parsed_variables = self.plan.eval(self._with.initial_values)
            match parsed_variables:
                case FunctionResults():
                    return parsed_variables.evaluations.variables
                case np.ndarray() | list():
                    return np.asarray(parsed_variables, dtype=np.float64)
                case None:
                    return None
                case _:
                    msg = f"`{self._with.initial_values}` does not contain variables."
                    raise ValueError(msg)
        return None
