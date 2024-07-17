"""This module defines the base classes for workflow objects.

Workflow objects can be added via the plugin mechanism to implement additional
workflow functionality. This is done by creating a plugin class that derives
from the [`WorkflowPlugin`][ropt.plugins.workflow.base.WorkflowPlugin] class. It
needs to define a [`create`][ropt.plugins.workflow.base.WorkflowPlugin].create
method that generates the workflow objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.workflow import ContextConfig, StepConfig
    from ropt.results import FunctionResults, Results
    from ropt.workflow import ContextUpdate, OptimizerContext, Workflow


class ContextObj(ABC):
    """Base class for workflow context objects."""

    def __init__(self, config: ContextConfig, workflow: Workflow) -> None:
        """Initialize the context object.

        The `config` and `workflow` arguments are accessible as `context_config`
        and `workflow` properties.

        Args:
            config:   The configuration of the context object
            workflow: The parent workflow that contains the object
        """
        self._context_config = config
        self._workflow = workflow

    @abstractmethod
    def update(self, value: ContextUpdate) -> None:
        """Update the value of the object.

        Args:
            value: The value used for the update.
        """

    def reset(self) -> None:  # noqa: B027
        """Resets the object to its initial state."""

    @property
    def context_config(self) -> ContextConfig:
        """Return the context object configuration.

        Returns:
            The configuration object.
        """
        return self._context_config

    @property
    def workflow(self) -> Workflow:
        """Return the workflow object.

        Returns:
            The workflow object.
        """
        return self._workflow

    def get_variable(self) -> Any:  # noqa: ANN401
        """Get a variable with the name equal to the context object ID.

        Returns:
            The value of the variable.
        """
        return self._workflow[self._context_config.id]

    def set_variable(self, value: Any) -> None:  # noqa: ANN401
        """Set a variable with the name equal to the context object ID.

        Args:
            value: The value
        """
        self._workflow[self._context_config.id] = value


class WorkflowStep(ABC):
    """Base class for workflow steps."""

    def __init__(self, config: StepConfig, workflow: Workflow) -> None:
        """Initialize the workflow object.

        Args:
            config:   The configuration of the workflow object
            workflow: The parent workflow
        """
        self._step_config = config
        self._workflow = workflow

    @property
    def step_config(self) -> StepConfig:
        """Return the step object config.

        Returns:
            The configuration object.
        """
        return self._step_config

    @property
    def workflow(self) -> Workflow:
        """Return the workflow object.

        Returns:
            The workflow object.
        """
        return self._workflow

    @abstractmethod
    def run(self) -> bool:
        """Run the step object.

        Returns:
            `True` if a user abort occurred.
        """


class OptimizerStep(WorkflowStep):
    """Base class for optimizer steps."""

    @abstractmethod
    def start_evaluation(self) -> None:
        """Called before the optimizer starts an evaluation."""

    @abstractmethod
    def finish_evaluation(self, results: Tuple[Results, ...]) -> None:
        """Called after the optimizer finishes an evaluation.

        Args:
            results: The results produced by the evaluation.
        """

    @abstractmethod
    def run_nested_workflow(
        self, variables: NDArray[np.float64]
    ) -> Tuple[Optional[FunctionResults], bool]:
        """Run a  nested workflow.

        Args:
            variables: variables to set in the nested workflow.

        Returns:
            The variables generated by the nested workflow.
        """


class WorkflowPlugin(Plugin):
    """The abstract base class for workflow plugins."""

    @abstractmethod
    def create(
        self, config: Union[ContextConfig, StepConfig], context: OptimizerContext
    ) -> Union[ContextObj, WorkflowStep]:
        """Create the workflow object.

        Args:
            config:  The configuration of the workflow object
            context: The context in which the workflow operates
        """
