"""This module implements the default optimizer step."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.config.plan import PlanConfig  # noqa: TCH001
from ropt.config.utils import Array1D, ItemOrSet, ItemOrTuple  # noqa: TCH001
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.optimization import EnsembleOptimizer
from ropt.plan import Event, Plan, RunStep
from ropt.results import FunctionResults
from ropt.utils.scaling import scale_variables

if sys.version_info >= (3, 11):
    pass
else:
    pass

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.plan import RunStepConfig
    from ropt.results import Results

MetaDataType = Dict[str, Union[int, float, bool, str]]


class NestedPlanConfig(BaseModel):
    """Parameters needed by a nested plan.

    Attributes:
        plan:         The nested plan
        extra_inputs: Extra inputs passed to the plan
    """

    plan: PlanConfig
    extra_inputs: ItemOrTuple[Any] = ()


class DefaultOptimizerStepWith(BaseModel):
    """Parameters used by the default optimizer step.

    Optionally the initial variables to be used can be set from an results
    handler object.

    Attributes:
        config:              The optimizer configuration
        tags:   Tags to      add to the emitted events
        initial_values:      The initial values for the optimizer
        exit_code_var:       Name of the variable to store the exit code
        nested_optimization: Optional nested optimization plan configuration
    """

    config: str
    tags: ItemOrSet[str] = set()
    initial_values: Optional[Union[str, Array1D]] = None
    exit_code_var: Optional[str] = None
    nested_optimization: Optional[NestedPlanConfig] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultOptimizerStep(RunStep):
    """The default optimizer step."""

    def __init__(self, config: RunStepConfig, plan: Plan) -> None:
        """Initialize a default optimizer step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._with = (
            DefaultOptimizerStepWith.model_validate({"config": config.with_})
            if isinstance(config.with_, str)
            else DefaultOptimizerStepWith.model_validate(config.with_)
        )
        self._enopt_config: EnOptConfig

    def run(self) -> None:
        """Run the optimizer step.

        Returns:
            Whether a user abort occurred.
        """
        config = self.plan.eval(self._with.config)
        if not isinstance(config, (dict, EnOptConfig)):
            msg = "No valid EnOpt configuration provided"
            raise TypeError(msg)
        self._enopt_config = EnOptConfig.model_validate(config)

        self.plan.emit_event(
            Event(
                event_type=EventType.START_OPTIMIZER_STEP,
                config=self._enopt_config,
                tags=self._with.tags,
            )
        )

        assert self.plan.optimizer_context.rng is not None
        ensemble_evaluator = EnsembleEvaluator(
            self._enopt_config,
            self.plan.optimizer_context.evaluator,
            self.plan.optimizer_context.result_id_iter,
            self.plan.optimizer_context.rng,
            self.plan.plugin_manager,
        )

        variables = self._get_variables(self._enopt_config)
        ensemble_optimizer = EnsembleOptimizer(
            enopt_config=self._enopt_config,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self.plan.plugin_manager,
            nested_optimizer=(
                self._run_nested_plan
                if self._with.nested_optimization is not None
                else None
            ),
            signal_evaluation=self._signal_evaluation,
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

        self.plan.emit_event(
            Event(
                event_type=EventType.FINISHED_OPTIMIZER_STEP,
                config=self._enopt_config,
                tags=self._with.tags,
                exit_code=exit_code,
            )
        )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

    def _signal_evaluation(self, results: Optional[Tuple[Results, ...]] = None) -> None:
        """Called before and after the optimizer finishes an evaluation.

        Before the evaluation starts, this method is called with the `results`
        argument set to `None`. When an evaluation is has finished, it is called
        with `results` set to the results of the evaluation.

        Args:
            results: The results produced by the evaluation.
        """
        if results is None:
            self.plan.emit_event(
                Event(
                    event_type=EventType.START_EVALUATION,
                    config=self._enopt_config,
                    tags=self._with.tags,
                )
            )
        else:
            self.plan.emit_event(
                Event(
                    event_type=EventType.FINISHED_EVALUATION,
                    config=self._enopt_config,
                    results=results,
                    tags=self._with.tags,
                )
            )

    def _run_nested_plan(
        self, variables: NDArray[np.float64]
    ) -> Tuple[Optional[FunctionResults], bool]:
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

    def _get_variables(self, config: EnOptConfig) -> NDArray[np.float64]:
        if self._with.initial_values is not None:
            parsed_variables = self.plan.eval(self._with.initial_values)
            if isinstance(parsed_variables, FunctionResults):
                return (
                    parsed_variables.evaluations.variables
                    if parsed_variables.evaluations.scaled_variables is None
                    else parsed_variables.evaluations.scaled_variables
                )
            if isinstance(parsed_variables, np.ndarray):
                scaled_variables = scale_variables(config, parsed_variables, axis=-1)
                return (
                    parsed_variables if scaled_variables is None else scaled_variables
                )
            if parsed_variables is not None:
                msg = f"`{self._with.initial_values} does not contain variables."
                raise ValueError(msg)
        return self._enopt_config.variables.initial_values
