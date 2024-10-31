"""This module defines the base classes for optimization plan plugins.

Optimization plan steps and result handlers can be added via the plugin
mechanism to implement additional functionality. This is done by creating a
plugin class that derives from the
[`PlanPlugin`][ropt.plugins.plan.base.PlanPlugin] class. It needs to define a
[`create`][ropt.plugins.plan.base.PlanPlugin.create] method that generates the
plan objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config.plan import PlanStepConfig, ResultHandlerConfig
    from ropt.plan import Event, OptimizerContext, Plan


class PlanPlugin(Plugin):
    """The abstract base class for plan plugins.

    Plan steps and result handlers are implemented through plugins that must derive
    from this base class. Plugins can be built-in, installed via a plugin mechanism,
    or loaded dynamically. During execution of an optimization plan, the
    required optimizer plugin will be located via the
    [`PluginManager`][ropt.plugins.PluginManager] and used to create
    [`PlanStep`][ropt.plugins.plan.base.PlanStep] or [`ResultHandler`][ropt.plugins.plan.base.ResultHandler] objects via its
    `create` function.
    """

    @abstractmethod
    def create(
        self,
        config: Union[ResultHandlerConfig, PlanStepConfig],
        context: OptimizerContext,
    ) -> Union[ResultHandler, PlanStep]:
        """Create a step or result handler.

        This factory function instantiates either a step or a result handler
        object based on the provided configuration. The configuration determines
        which type of object to return—either a step or a result handler.

        Args:
            config:  The configuration for the plan object.
            context: The context in which the plan operates.
        """


class PlanStep(ABC):
    """Base class for steps.

    steps loaded and created by the plugin manager must inherit from this
    abstract base class. It defines a `run` method that must be overridden to
    implement the specific functionality of each step. The class also provides
    default properties for accessing the configuration and the plan that
    executes the step.
    """

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize the step.

        The `config` and `plan` arguments are accessible as `step_config`
        and `plan` properties.

        Args:
            config: The configuration for this step.
            plan:   The parent plan that manages this step.
        """
        self._step_config = config
        self._plan = plan

    @property
    def step_config(self) -> PlanStepConfig:
        """Return the step object's configuration.

        Returns:
            The configuration object.
        """
        return self._step_config

    @property
    def plan(self) -> Plan:
        """Return the plan that executes the step.

        Returns:
            The plan object.
        """
        return self._plan

    @abstractmethod
    def run(self) -> None:
        """Execute the step object.

        This method must be overloaded to implement the functionality of the
        step.
        """


class ResultHandler(ABC):
    """Base class for result handler objects.

    Result handlers loaded and created by the plugin manager must inherit from
    this abstract base class. It defines a `handle_event` method that must be
    overridden to implement specific functionality and provides default
    properties for accessing the handler's configuration and the plan that runs
    it.
    """

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize a results handler.

        The `config` and `plan` arguments are accessible as `handler_config`
        and `plan` properties.

        Args:
            config: The configuration of the handler object
            plan:   The parent plan that contains the object
        """
        self._handler_config = config
        self._plan = plan

    @property
    def handler_config(self) -> ResultHandlerConfig:
        """Return the configuration of the handler object.

        Returns:
            The configuration object.
        """
        return self._handler_config

    @property
    def plan(self) -> Plan:
        """Return the associated plan that executes the handler.

        Returns:
            The plan object.
        """
        return self._plan

    @abstractmethod
    def handle_event(self, event: Event) -> Event:
        """Handle and propagate an event.

        This method must be overloaded to implement the functionality of the
        handler, based on the information passed via the event object. The
        handler may modify the event, which may be propagated further to other
        handlers or to callbacks that react to the event.

        Returns:
            The event, possibly modified.
        """
