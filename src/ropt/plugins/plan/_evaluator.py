"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.config.utils import Array2D  # noqa: TCH001
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted, PlanError
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults
from ropt.utils.scaling import scale_variables

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultEvaluatorStepWith(BaseModel):
    """Parameters used by the default optimizer step.

    Optionally the initial variables to be used can be set from an context
    object.

    Attributes:
        config:   ID of the context object that contains the optimizer configuration
        values:   Values to evaluate at
    """

    config: str
    tags: Set[str] = set()
    values: Optional[Union[str, Array2D]] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultEvaluatorStep(PlanStep):
    """The default evaluator step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default evaluator step.

        Args:
            config: The configuration of the step
            plan:   The optimization plan that runs this step
        """
        super().__init__(config, plan)

        self._with = (
            DefaultEvaluatorStepWith.model_validate({"config": config.with_})
            if isinstance(config.with_, str)
            else DefaultEvaluatorStepWith.model_validate(config.with_)
        )

    def run(self) -> None:
        """Run the evaluator step."""
        config = self.plan.parse_value(self._with.config)
        if not isinstance(config, (dict, EnOptConfig)):
            msg = "No valid EnOpt configuration provided"
            raise PlanError(msg, step_name=self.step_config.name)
        self._enopt_config = EnOptConfig.model_validate(config)

        self.plan.emit_event(
            EventType.START_EVALUATOR_STEP,
            self._enopt_config,
            tags=self._with.tags,
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

        self.plan.emit_event(
            EventType.FINISHED_EVALUATOR_STEP,
            self._enopt_config,
            results=results,
            tags=self._with.tags,
            exit_code=exit_code,
            step_name=self.step_config.name,
        )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

    def _get_variables(self, config: EnOptConfig) -> NDArray[np.float64]:
        if self._with.values is not None:  # noqa: PD011
            parsed_variables = self.plan.parse_value(self._with.values)
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
                msg = f"`{self._with.values} does not contain variables."  # noqa: PD011
                raise PlanError(msg, step_name=self.step_config.name)
        return self._enopt_config.variables.initial_values
