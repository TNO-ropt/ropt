"""Defines Base Classes for Optimization Plan Components and Plugins."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.plan import Event, Plan


class PlanHandlerPlugin(Plugin):
    """Abstract Base Class for Plan Handler Plugins.

    This class defines the interface for plugins responsible for creating
    [`PlanHandler`][ropt.plugins.plan.base.PlanHandler] instances within an
    optimization plan ([`Plan`][ropt.plan.Plan]).

    During plan setup, the [`PluginManager`][ropt.plugins.PluginManager]
    identifies the appropriate handler plugin based on a requested name and
    uses its `create` class method to instantiate the actual `PlanHandler`
    object that will process events during plan execution.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        **kwargs: Any,  # noqa: ANN401
    ) -> PlanHandler:
        """Create a PlanHandler instance.

        This abstract class method serves as a factory for creating concrete
        [`PlanHandler`][ropt.plugins.plan.base.PlanHandler] objects. Plugin
        implementations must override this method to return an instance of
        their specific `PlanHandler` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method
        when a plan requests a handler provided by this plugin via
        [`Plan.add_handler`][ropt.plan.Plan.add_handler].

        The `name` argument specifies the requested handler, potentially in the
        format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created handler if the
        plugin supports multiple handler types.

        Any additional keyword arguments (`kwargs`) passed during the
        [`Plan.add_handler`][ropt.plan.Plan.add_handler] call are forwarded here,
        allowing for custom configuration of the handler instance.

        Args:
            name:   The requested handler name (potentially plugin-specific).
            plan:   The parent [`Plan`][ropt.plan.Plan] instance.
            kwargs: Additional arguments for custom configuration.

        Returns:
            An initialized instance of a `PlanHandler` subclass.
        """


class PlanStepPlugin(Plugin):
    """Abstract base class for plugins that create PlanStep instances.

    This class defines the interface for plugins that act as factories for
    [`PlanStep`][ropt.plugins.plan.base.PlanStep] objects.

    The [`PluginManager`][ropt.plugins.PluginManager] uses the `create` class
    method of these plugins to instantiate `PlanStep` objects when they are
    added to an optimization [`Plan`][ropt.plan.Plan] via
    [`Plan.add_step`][ropt.plan.Plan.add_step].
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        **kwargs: Any,  # noqa: ANN401
    ) -> PlanStep:
        """Create a PlanStep instance.

        This abstract class method serves as a factory for creating concrete
        [`PlanStep`][ropt.plugins.plan.base.PlanStep] objects. Plugin
        implementations must override this method to return an instance of
        their specific `PlanStep` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method
        when a plan requests a step provided by this plugin via
        [`Plan.add_step`][ropt.plan.Plan.add_step].

        The `name` argument specifies the requested step, potentially in the
        format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created step if the
        plugin supports multiple step types.

        Any additional keyword arguments (`kwargs`) passed during the
        [`Plan.add_step`][ropt.plan.Plan.add_step] call are forwarded here,
        allowing for custom configuration of the step instance.

        Args:
            name:   The requested step name (potentially plugin-specific).
            plan:   The parent [`Plan`][ropt.plan.Plan] instance.
            kwargs: Additional arguments for custom configuration.

        Returns:
            An initialized instance of a `PlanStep` subclass.
        """


class PlanStep(ABC):
    """Abstract Base Class for Optimization Plan Steps.

    This class defines the fundamental interface for all executable steps within
    an optimization [`Plan`][ropt.plan.Plan]. Concrete step implementations,
    which perform specific actions like running an optimizer or evaluating
    functions, must inherit from this base class.

    `PlanStep` objects are typically created by corresponding
    [`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin] factories, which
    are managed by the [`PluginManager`][ropt.plugins.PluginManager]. Once
    instantiated and added to a `Plan`, their
    [`run`][ropt.plugins.plan.base.PlanStep.run] method is called by the plan
    during execution.

    Each `PlanStep` instance has a unique [`id`][ropt.plugins.plan.base.PlanStep.id]
    and maintains a reference to its parent
    [`plan`][ropt.plugins.plan.base.PlanStep.plan].

    Subclasses must implement the abstract
    [`run`][ropt.plugins.plan.base.PlanStep.run] method to define the step's
    specific behavior.
    """

    def __init__(self, plan: Plan) -> None:
        """Initialize the PlanStep.

        Associates the step with its parent [`Plan`][ropt.plan.Plan] and assigns
        a unique ID. The parent plan is accessible via the `plan` property.

        Args:
            plan: The [`Plan`][ropt.plan.Plan] instance that owns this step.
        """
        self.__stored_plan = plan
        self.__id: uuid.UUID = uuid.uuid4()

    @property
    def id(self) -> uuid.UUID:
        """Return the unique identifier of the handler.

        Returns:
            A UUID object representing the unique identifier of the handler.
        """
        return self.__id

    @property
    def plan(self) -> Plan:
        """Return the parent plan associated with this step.

        Returns:
            The [`Plan`][ropt.plan.Plan] object that owns and executes this step.
        """
        return self.__stored_plan

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Execute the logic defined by this plan step.

        This abstract method must be implemented by concrete `PlanStep`
        subclasses to define the specific action the step performs within the
        optimization [`Plan`][ropt.plan.Plan].

        The `Plan` object calls this method during its execution sequence,
        passing any arguments provided when the step was invoked via
        [`Plan.run_step`][ropt.plan.Plan.run_step]. The return value and type
        can vary depending on the specific step implementation.

        Args:
            args:   Positional arguments passed from `Plan.run_step`.
            kwargs: Keyword arguments passed from `Plan.run_step`.

        Returns:
            The result of the step's execution, if any.
        """


class PlanHandler(ABC):
    """Abstract Base Class for Optimization Plan Result Handlers.

    This class defines the fundamental interface for all result handlers within
    an optimization [`Plan`][ropt.plan.Plan]. Concrete handler implementations,
    which process events emitted during plan execution (e.g., tracking results,
    storing data, logging), must inherit from this base class.

    `PlanHandler` objects are typically created by corresponding
    [`PlanHandlerPlugin`][ropt.plugins.plan.base.PlanHandlerPlugin] factories,
    managed by the [`PluginManager`][ropt.plugins.PluginManager]. Once
    instantiated and added to a `Plan`, their
    [`handle_event`][ropt.plugins.plan.base.PlanHandler.handle_event] method is
    called by the plan whenever an [`Event`][ropt.plan.Event] is emitted.

    Handlers can also store state using dictionary-like access (`[]`), allowing
    them to accumulate information or make data available to subsequent steps
    or handlers within the plan.

    Each `PlanHandler` instance has a unique
    [`id`][ropt.plugins.plan.base.PlanHandler.id] and maintains a reference to
    its parent [`plan`][ropt.plugins.plan.base.PlanHandler.plan].

    Subclasses must implement the abstract
    [`handle_event`][ropt.plugins.plan.base.PlanHandler.handle_event] method to
    define their specific event processing logic.
    """

    def __init__(self, plan: Plan) -> None:
        """Initialize the PlanHandler.

        Associates the handler with its parent [`Plan`][ropt.plan.Plan], assigns
        a unique ID, and initializes an internal dictionary for storing state.
        The parent plan is accessible via the `plan` property.

        Args:
            plan: The [`Plan`][ropt.plan.Plan] instance that owns this handler.
        """
        self.__stored_plan = plan
        self.__stored_values: dict[str, Any] = {}
        self.__id: uuid.UUID = uuid.uuid4()

    @property
    def id(self) -> uuid.UUID:
        """Return the unique identifier (UUID) of this handler instance.

        Returns:
            The unique UUID object for this handler.
        """
        return self.__id

    @property
    def plan(self) -> Plan:
        """Return the parent plan associated with this handler.

        Returns:
            The [`Plan`][ropt.plan.Plan] object that owns this handler.
        """
        return self.__stored_plan

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Process an event emitted by the optimization plan.

        This abstract method must be implemented by concrete `PlanHandler`
        subclasses. It defines the handler's core logic for reacting to
        [`Event`][ropt.plan.Event] objects emitted during the execution of the
        parent [`Plan`][ropt.plan.Plan].

        Implementations should inspect the `event` object (its `event_type` and
        `data`) and perform actions accordingly, such as storing results,
        logging information, or updating internal state.

        Args:
            event: The [`Event`][ropt.plan.Event] object containing details
                   about what occurred in the plan.
        """

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Retrieve a value from the handler's internal state.

        This method enables dictionary-like access (`handler[key]`) to the
        values stored within the handler's internal state dictionary. This
        allows handlers to store and retrieve data accumulated during plan
        execution.

        Args:
            key: The string key identifying the value to retrieve.

        Returns:
            The value associated with the specified key.

        Raises:
            AttributeError: If the provided `key` does not exist in the
                            handler's stored values.
        """
        if key in self.__stored_values:
            return self.__stored_values[key]
        msg = f"Unknown plan variable: `{key}`"
        raise AttributeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Store or update a value in the handler's internal state.

        This method enables dictionary-like assignment (`handler[key] = value`)
        to store arbitrary data within the handler's internal state dictionary.
        This allows handlers to accumulate information or make data available to
        other components of the plan.

        The key must be a valid Python identifier.

        Args:
            key:   The string key identifying the value to store (must be an identifier).
            value: The value to associate with the key.

        Raises:
            AttributeError: If the provided `key` is not a valid identifier.
        """
        if not key.isidentifier():
            msg = f"Not a valid key: `{key}`"
            raise AttributeError(msg)
        self.__stored_values[key] = value
