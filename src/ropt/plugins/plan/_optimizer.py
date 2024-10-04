"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.config.plan import PlanConfig  # noqa: TCH001
from ropt.config.utils import Array1D  # noqa: TCH001
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import PlanError
from ropt.plan import ContextUpdateResults, EnsembleOptimizer, MetaDataType, Plan
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults
from ropt.utils.scaling import scale_variables

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.plan import StepConfig
    from ropt.results import Results


class DefaultNestedPlan(BaseModel):
    """Parameters used by the nested optimizer step.

    Attributes:
        plan:        Optional nested plan to run during optimization
        initial_var: Name of the variable providing initial_values to the nested plan
        results_var: The name of the variable in the plan containing the result
    """

    plan: PlanConfig
    initial_var: str
    results_var: str


class DefaultOptimizerStepWith(BaseModel):
    """Parameters used by the default optimizer step.

    Optionally the initial variables to be used can be set from a context object.

    Attributes:
        config:         ID of the context object that contains the optimizer configuration
        update:         List of the objects that are notified of new results
        initial_values: The initial values for the optimizer
        exit_code_var:  Name of the variable to store the exit code
        nested_plan:    Optional nested plan configuration
    """

    config: str
    update: List[str] = []
    initial_values: Optional[Union[str, Array1D]] = None
    exit_code_var: Optional[str] = None
    nested_plan: Optional[DefaultNestedPlan] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultOptimizerStep(PlanStep):
    """The default optimizer step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
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

    def run(self) -> bool:
        """Run the optimizer step.

        Returns:
            Whether a user abort occurred.
        """
        config = self.plan.parse_value(self._with.config)
        if not isinstance(config, (dict, EnOptConfig)):
            msg = "No valid EnOpt configuration provided"
            raise PlanError(msg, step_name=self.step_config.name)
        self._enopt_config = EnOptConfig.model_validate(config)

        self.plan.optimizer_context.events.emit(
            EventType.START_OPTIMIZER_STEP,
            self._enopt_config,
            step_name=self.step_config.name,
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
        exit_code = EnsembleOptimizer(
            enopt_config=self._enopt_config,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self.plan.plugin_manager,
            nested_optimizer=self._run_nested_plan,
            signal_evaluation=self._signal_evaluation,
        ).start(variables)

        if self._with.exit_code_var is not None:
            self.plan[self._with.exit_code_var] = exit_code

        self.plan.optimizer_context.events.emit(
            EventType.FINISHED_OPTIMIZER_STEP,
            self._enopt_config,
            exit_code=exit_code,
            step_name=self.step_config.name,
        )

        return exit_code == OptimizerExitCode.USER_ABORT

    def _signal_evaluation(self, results: Optional[Tuple[Results, ...]] = None) -> None:
        """Called before and after the optimizer finishes an evaluation.

        Before the evaluation starts, this method is called with the `results`
        argument set to `None`. When an evaluation is has finished, it is called
        with `results` set to the results of the evaluation.

        Args:
            results: The results produced by the evaluation.
        """
        if results is None:
            self.plan.optimizer_context.events.emit(
                EventType.START_EVALUATION,
                self._enopt_config,
                step_name=self.step_config.name,
            )
        else:
            metadata = self._get_metadata(add_step_name=True)
            for item in results:
                item.metadata = metadata

            for obj_id in self._with.update:
                self.plan.update_context(
                    obj_id,
                    ContextUpdateResults(
                        step_name=self.step_config.name,
                        results=results,
                    ),
                )
            self.plan.optimizer_context.events.emit(
                EventType.FINISHED_EVALUATION,
                self._enopt_config,
                results=results,
                step_name=self.step_config.name,
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
        if self._with.nested_plan is None:
            return None, False
        plan = self.plan.spawn(self._with.nested_plan.plan)
        plan[self._with.nested_plan.initial_var] = variables
        aborted = plan.run()
        return plan[self._with.nested_plan.results_var], aborted

    def _get_variables(self, config: EnOptConfig) -> NDArray[np.float64]:
        if self._with.initial_values is not None:
            parsed_variables = self.plan.parse_value(self._with.initial_values)
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
                raise PlanError(msg, step_name=self.step_config.name)
        return self._enopt_config.variables.initial_values

    def _get_metadata(self, *, add_step_name: bool) -> MetaDataType:
        metadata = self.plan.optimizer_context.metadata
        if add_step_name and self.step_config.name is not None:
            metadata["step_name"] = self.step_config.name
        return metadata
