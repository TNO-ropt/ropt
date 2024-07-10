"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.config.utils import Array2D  # noqa: TCH001
from ropt.enums import OptimizerExitCode
from ropt.evaluator import EnsembleEvaluator
from ropt.exceptions import OptimizationAborted, WorkflowError
from ropt.plugins.workflow.base import WorkflowStep
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.workflow import StepConfig
    from ropt.workflow import Workflow


class DefaultEvaluatorStepWith(BaseModel):
    """Parameters used by the default optimizer step.

    Optionally the initial variables to be used can be set from an context
    object.

    Attributes:
        config:   ID of the context object that contains the optimizer configuration
        update:   List of the objects that are notified of new results
        values:   Values to evaluate at
        metadata: Metadata to set in the results
    """

    config: str
    update: List[str] = []
    values: Optional[Union[str, Array2D]] = None
    metadata: Dict[str, str] = {}

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultEvaluatorStep(WorkflowStep):
    """The default evaluator step."""

    def __init__(self, config: StepConfig, workflow: Workflow) -> None:
        """Initialize a default evaluator step.

        Args:
            config:   The configuration of the step
            workflow: The workflow that runs this step
        """
        super().__init__(config, workflow)

        self._with = (
            DefaultEvaluatorStepWith.model_validate({"config": config.with_})
            if isinstance(config.with_, str)
            else DefaultEvaluatorStepWith.model_validate(config.with_)
        )

    def run(self) -> bool:
        """Run the evaluator step.

        Returns:
            Whether a user abort occurred.
        """
        config = self.workflow.parse_value(self._with.config)
        if not isinstance(config, (dict, EnOptConfig)):
            msg = "No valid EnOpt configuration provided"
            raise WorkflowError(msg, step_name=self.step_config.name)
        self._enopt_config = EnOptConfig.model_validate(config)

        assert self.workflow.optimizer_context.rng is not None
        ensemble_evaluator = EnsembleEvaluator(
            self._enopt_config,
            self.workflow.optimizer_context.evaluator,
            self.workflow.optimizer_context.result_id_iter,
            self.workflow.optimizer_context.rng,
            self.workflow.plugin_manager,
        )

        variables = self._get_variables()
        exit_code = OptimizerExitCode.EVALUATION_STEP_FINISHED
        try:
            results = ensemble_evaluator.calculate(
                variables, compute_functions=True, compute_gradients=False
            )
        except OptimizationAborted as exc:
            exit_code = exc.exit_code

        for item in results:
            if self.step_config.name is not None:
                item.metadata["step_name"] = self.step_config.name
            for key, expr in self._with.metadata.items():
                item.metadata[key] = self.workflow.parse_value(expr)

        for obj_id in self._with.update:
            self.workflow.update_context(obj_id, results)

        assert results
        assert isinstance(results[0], FunctionResults)
        if results[0].functions is None:
            exit_code = OptimizerExitCode.TOO_FEW_REALIZATIONS

        return exit_code == OptimizerExitCode.USER_ABORT

    def _get_variables(self) -> NDArray[np.float64]:
        if self._with.values is not None:  # noqa: PD011
            parsed_variables = self.workflow.parse_value(self._with.values)
            if isinstance(parsed_variables, FunctionResults):
                return parsed_variables.evaluations.variables
            if isinstance(parsed_variables, np.ndarray):
                return parsed_variables
            if parsed_variables is not None:
                msg = f"`{self._with.values} does not contain variables."  # noqa: PD011
                raise WorkflowError(msg, step_name=self.step_config.name)
        return self._enopt_config.variables.initial_values
