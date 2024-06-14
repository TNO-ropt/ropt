"""This module implements the default workflow plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from typing import Dict, Final, Type, Union

from ropt.config.workflow import ContextConfig, StepConfig  # noqa: TCH001
from ropt.plugins.workflow.base import ContextObj, WorkflowStep  # noqa: TCH001
from ropt.workflow import Workflow  # noqa: TCH001

from ._enopt_config import DefaultEnOptConfigContext
from ._evaluator import DefaultEvaluatorStep
from ._optimizer import DefaultOptimizerStep
from ._repeat import DefaultRepeatStep
from ._reset_context import DefaultResetContextStep
from ._results_callback import DefaultResultsCallbackContext
from ._results_tracker import DefaultResultsTrackerContext
from ._update_context import DefaultUpdateContextStep
from .base import WorkflowPlugin

_CONTEXT_OBJECTS: Final[Dict[str, Type[ContextObj]]] = {
    "results_callback": DefaultResultsCallbackContext,
    "enopt_config": DefaultEnOptConfigContext,
    "results_tracker": DefaultResultsTrackerContext,
}

_STEP_OBJECTS: Final[Dict[str, Type[WorkflowStep]]] = {
    "evaluator": DefaultEvaluatorStep,
    "optimizer": DefaultOptimizerStep,
    "repeat": DefaultRepeatStep,
    "reset_context": DefaultResetContextStep,
    "update_context": DefaultUpdateContextStep,
}


class DefaultWorkflowPlugin(WorkflowPlugin):
    """Default workflow plugin class."""

    @singledispatchmethod
    def create(  # type: ignore
        self,
        config: Union[ContextConfig, StepConfig],  # noqa: ARG002
        workflow: Workflow,  # noqa: ARG002
    ) -> Union[ContextObj, WorkflowStep]:
        """Initialize the workflow plugin.

        See the [ropt.plugins.workflow.base.WorkflowPlugin][] abstract base class.

        # noqa
        """
        msg = "Workflow config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_context(self, config: ContextConfig, workflow: Workflow) -> ContextObj:
        _, _, name = config.init.lower().rpartition("/")
        obj = _CONTEXT_OBJECTS.get(name)
        if obj is not None:
            return obj(config, workflow)

        msg = f"Unknown context object type: {config.init}"
        raise TypeError(msg)

    @create.register
    def _create_step(self, config: StepConfig, workflow: Workflow) -> WorkflowStep:
        _, _, step_name = config.run.lower().rpartition("/")
        step_obj = _STEP_OBJECTS.get(step_name)
        if step_obj is not None:
            return step_obj(config, workflow)

        msg = f"Unknown step type: {config.run}"
        raise TypeError(msg)

    def is_supported(self, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return (method.lower() in _CONTEXT_OBJECTS) or (method.lower() in _STEP_OBJECTS)
