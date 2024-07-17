"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.config.utils import Array1D  # noqa: TCH001
from ropt.config.workflow import WorkflowConfig  # noqa: TCH001
from ropt.enums import EventType, OptimizerExitCode
from ropt.evaluator import EnsembleEvaluator
from ropt.exceptions import WorkflowError
from ropt.plugins.workflow.base import OptimizerStep
from ropt.results import FunctionResults
from ropt.workflow import ContextUpdateResults, Optimizer, Workflow

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.workflow import StepConfig
    from ropt.results import Results


class DefaultNestedWorkflow(BaseModel):
    """Parameters used by the nested optimizer step.

    Attributes:
        workflow:    Optional nested workflow to run during optimization
        initial_var: Name of the variable providing initial_values to the nested workflow
        results_var: The name of the variable in the workflow containing the result
    """

    workflow: WorkflowConfig
    initial_var: str
    results_var: str


class DefaultOptimizerStepWith(BaseModel):
    """Parameters used by the default optimizer step.

    Optionally the initial variables to be used can be set from a context object.

    Attributes:
        config:          ID of the context object that contains the optimizer configuration
        update:          List of the objects that are notified of new results
        initial_values:  The initial values for the optimizer
        metadata:        Metadata to set in the results
        exit_code_var:   Name of the variable to store the exit code
        nested_workflow: Optional nested workflow configuration
    """

    config: str
    update: List[str] = []
    initial_values: Optional[Union[str, Array1D]] = None
    metadata: Dict[str, Union[int, float, bool, str]] = {}
    exit_code_var: Optional[str] = None
    nested_workflow: Optional[DefaultNestedWorkflow] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultOptimizerStep(OptimizerStep):
    """The default optimizer step."""

    def __init__(self, config: StepConfig, workflow: Workflow) -> None:
        """Initialize a default optimizer step.

        Args:
            config:   The configuration of the step
            workflow: The workflow that runs this step
        """
        super().__init__(config, workflow)

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
        exit_code = Optimizer(
            enopt_config=self._enopt_config,
            optimizer_step=self,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self.workflow.plugin_manager,
        ).start(variables)

        if self._with.exit_code_var is not None:
            self.workflow[self._with.exit_code_var] = exit_code

        self.workflow.optimizer_context.events.emit(
            event_type=EventType.FINISHED_OPTIMIZER_STEP,
            config=self._enopt_config,
            exit_code=exit_code,
        )

        return exit_code == OptimizerExitCode.USER_ABORT

    def start_evaluation(self) -> None:
        """Call before an evaluation is started."""

    def finish_evaluation(self, results: Tuple[Results, ...]) -> None:
        """Called after the optimizer finishes an evaluation.

        Args:
            results: The results produced by the evaluation.
        """
        for item in results:
            if self.step_config.name is not None:
                item.metadata["step_name"] = self.step_config.name
            for key, expr in self._with.metadata.items():
                item.metadata[key] = self.workflow.parse_value(expr)

        for obj_id in self._with.update:
            self.workflow.update_context(
                obj_id,
                ContextUpdateResults(
                    step_name=self.step_config.name,
                    results=results,
                ),
            )
        self.workflow.optimizer_context.events.emit(
            event_type=EventType.FINISHED_EVALUATION,
            config=self._enopt_config,
            results=results,
        )

    def run_nested_workflow(
        self, variables: NDArray[np.float64]
    ) -> Tuple[Optional[FunctionResults], bool]:
        """Run a  nested workflow.

        Args:
            variables: variables to set in the nested workflow.

        Returns:
            The variables generated by the nested workflow.
        """
        if self._with.nested_workflow is None:
            return None, False
        workflow = self.workflow.spawn(self._with.nested_workflow.workflow)
        workflow[self._with.nested_workflow.initial_var] = variables
        aborted = workflow.run()
        return workflow[self._with.nested_workflow.results_var], aborted

    def _get_variables(self) -> NDArray[np.float64]:
        if self._with.initial_values is not None:
            parsed_variables = self.workflow.parse_value(self._with.initial_values)
            if isinstance(parsed_variables, FunctionResults):
                return parsed_variables.evaluations.variables
            if isinstance(parsed_variables, np.ndarray):
                return parsed_variables
            if parsed_variables is not None:
                msg = f"`{self._with.initial_values} does not contain variables."
                raise WorkflowError(msg, step_name=self.step_config.name)
        return self._enopt_config.variables.initial_values
