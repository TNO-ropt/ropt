"""Defines Base Classes for Optimization Plan Components and Plugins."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult
    from ropt.plan import Event, Plan


class EventHandlerPlugin(Plugin):
    """Abstract Base Class for Plan Handler Plugins.

    This class defines the interface for plugins responsible for creating
    [`EventHandler`][ropt.plugins.plan.base.EventHandler] instances within an
    optimization plan ([`Plan`][ropt.plan.Plan]).

    During plan setup, the [`PluginManager`][ropt.plugins.PluginManager]
    identifies the appropriate event handler plugin based on a requested name
    and uses its `create` class method to instantiate the actual `EventHandler`
    object that will process events during plan execution.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        tags: set[str] | None = None,
        sources: set[PlanComponent | str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> EventHandler:
        """Create a EventHandler instance.

        This abstract class method serves as a factory for creating concrete
        [`EventHandler`][ropt.plugins.plan.base.EventHandler] objects. Plugin
        implementations must override this method to return an instance of their
        specific `EventHandler` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        a plan requests an event handler provided by this plugin via
        [`Plan.add_event_handler`][ropt.plan.Plan.add_event_handler].

        The `name` argument specifies the requested event handler, potentially
        in the format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created event handler if
        the plugin supports multiple event handler types.

        The optional `tags` argument assigns the given string tags to the event
        handler. Similar to its `id`, the `tags` can be used for identification.
        However, unlike an `id`, a tag does not need to be unique, allowing
        multiple components to be grouped under the same tag.

        The `sources` parameter acts as a filter, determining which plan steps
        this event handler should listen to. It should be a set containing the
        `PlanStep` instances whose event you want to receive. When an event is
        received, this event handler checks if the step that emitted the event
        (`event.source`) is present in the `sources` set. If `sources` is
        `None`, events from all sources will be processed.

        Any additional keyword arguments (`kwargs`) passed during the
        [`Plan.add_event_handler`][ropt.plan.Plan.add_event_handler] call are
        forwarded here, allowing for custom configuration of the event handler
        instance.

        Args:
            name:    The requested event handler name (potentially plugin-specific).
            plan:    The parent [`Plan`][ropt.plan.Plan] instance.
            tags:    Optional tags
            sources: The steps whose events should be processed.
            kwargs:  Additional arguments for custom configuration.

        Returns:
            An initialized instance of a `EventHandler` subclass.
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
        tags: set[str] | None = None,
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

        The optional `tags` argument assigns the given string tags to the plan
        step. Similar to its `id`, the `tags` can be used for identification.
        However, unlike an `id`, a tag does not need to be unique, allowing
        multiple components to be grouped under the same tag.

        Any additional keyword arguments (`kwargs`) passed during the
        [`Plan.add_step`][ropt.plan.Plan.add_step] call are forwarded here,
        allowing for custom configuration of the step instance.

        Args:
            name:   The requested step name (potentially plugin-specific).
            plan:   The parent [`Plan`][ropt.plan.Plan] instance.
            tags:   Optional tags
            kwargs: Additional arguments for custom configuration.

        Returns:
            An initialized instance of a `PlanStep` subclass.
        """


class EvaluatorPlugin(Plugin):
    """Abstract base class for evaluator plugins.

    This class defines the interface for plugins responsible for creating
    plan-aware [`Evaluator`][ropt.plugins.plan.base.Evaluator] instances within
    an optimization plan ([`Plan`][ropt.plan.Plan]).

    During plan setup, the [`PluginManager`][ropt.plugins.PluginManager]
    identifies the appropriate evaluator plugin based on a requested name and
    uses its `create` class method to instantiate the actual `Evaluator` object
    that will perform evaluations during plan execution.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        tags: set[str] | None = None,
        clients: set[PlanComponent | str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Evaluator:
        """Create an Evaluator instance.

        This abstract class method serves as a factory for creating concrete
        [`Evaluator`][ropt.plugins.plan.base.Evaluator] objects. Plugin
        implementations must override this method to return an instance of their
        specific `Evaluator` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        an evaluator provided by this plugin is requested.

        The `name` argument specifies the requested evaluator, potentially
        in the format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created evaluator if
        the plugin supports multiple evaluator types.

        The optional `tags` argument assigns the given string tags to the
        evaluator. Similar to its `id`, the `tags` can be used for
        identification. However, unlike an `id`, a tag does not need to be
        unique, allowing multiple components to be grouped under the same tag.

        The `clients` parameter acts as a filter, determining which plan steps
        this evaluator should serve. It should be a set containing the
        `PlanStep` instances that should be handled. When an evaluation is
        requested, this evaluator checks if the step  is present in the `client`
        set.

        Args:
            name:    The requested evaluator name (potentially plugin-specific).
            plan:    The parent [`Plan`][ropt.plan.Plan] instance.
            tags:    Optional tags
            clients: The clients that should be served by this evaluator.
            kwargs:  Additional arguments for custom configuration.

        Returns:
            An initialized instance of an `Evaluator` subclass.
        """


class PlanComponent:
    """Base class for components that are part of an optimization plan.

    This class provides common functionality for components like steps, event
    handlers, and evaluators that are managed within a
    [`Plan`][ropt.plan.Plan].

    Each `PlanComponent` is assigned a unique identifier (`id`) upon
    initialization, an optional tag (`tag`), and maintains a reference to its
    parent `plan`.

    """

    def __init__(self, plan: Plan, tags: set[str] | None) -> None:
        """Initialize the PlanComponent.

        This constructor is called by subclasses to set up common attributes. It
        stores a reference to the parent `plan`, an optional `tag`, and assigns
        a unique `id`.

        Args:
            plan: The parent [`Plan`][ropt.plan.Plan] instance.
            tags: Optional tags
        """
        self.__stored_plan = plan
        self.__tags = set() if tags is None else tags
        self.__id: uuid.UUID = uuid.uuid4()

    @property
    def id(self) -> uuid.UUID:
        """Return the unique identifier of the event handler.

        Returns:
            A UUID object representing the unique identifier of the event handler.
        """
        return self.__id

    @property
    def plan(self) -> Plan:
        """Return the parent plan associated with this event handler.

        Returns:
            The [`Plan`][ropt.plan.Plan] object that owns this event handler.
        """
        return self.__stored_plan

    @property
    def tags(self) -> set[str]:
        """Return the optional tags.

        Returns:
            The tags.
        """
        return self.__tags


class PlanStep(ABC, PlanComponent):
    """Abstract base class for optimization plan steps.

    This class defines the fundamental interface for all executable steps within
    an optimization [`Plan`][ropt.plan.Plan]. Concrete step implementations,
    which perform specific actions like running an optimizer or evaluating
    functions, must inherit from this base class.

    `PlanStep` objects are typically created by corresponding
    [`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin] factories, which
    are managed by the [`PluginManager`][ropt.plugins.PluginManager]. Once
    instantiated and added to a `Plan`, their
    [`run_step_from_plan`][ropt.plugins.plan.base.PlanStep.run_step_from_plan]
    method is called by the plan during execution. This is generally done
    indirectly by calling the [`run`][ropt.plugins.plan.base.PlanStep.run]
    method on the step object.

    Subclasses must implement the abstract
    [`run_step_from_plan`][ropt.plugins.plan.base.PlanStep.run_step_from_plan]
    method to define the step's specific behavior.
    """

    def __init__(self, plan: Plan, tags: set[str] | None = None) -> None:
        """Initialize the PlanStep.

        Associates the step with its parent [`Plan`][ropt.plan.Plan] and assigns
        a unique ID. The parent plan is accessible via the `plan` property.

        Args:
            plan: The [`Plan`][ropt.plan.Plan] instance that owns this step.
            tags: Optional tags
        """
        super().__init__(plan, tags)

    def run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Execute this plan step.

        This method initiates the execution of the current plan step. It
        delegates the actual execution to the parent
        [`Plan`][ropt.plan.Plan] object's
        [`run_step`][ropt.plan.Plan.run_step] method, passing itself
        (the step instance) along with any provided arguments.

        The parent `Plan` then calls the concrete
        [`run_step_from_plan`][ropt.plugins.plan.base.PlanStep.run_step_from_plan]
        method implemented by the subclass of this `PlanStep`. This allows the
        plan to do some bookkeeping, for instance to check if the plan was
        aborted.

        Args:
            *args:    Positional arguments to be passed to the step's specific `run_step` method.
            **kwargs: Keyword arguments to be passed to the step's specific `run_step` method.

        Returns:
            The result returned by the step's specific `run_step_from_plan` method.
        """
        return self.plan.run_step(self, *args, **kwargs)

    @abstractmethod
    def run_step_from_plan(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
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


class EventHandler(ABC, PlanComponent):
    """Abstract Base Class for Optimization Plan Result Handlers.

    This class defines the fundamental interface for all event handlers within
    an optimization [`Plan`][ropt.plan.Plan]. Concrete handler implementations,
    which process events emitted during plan execution (e.g., tracking results,
    storing data, logging), must inherit from this base class.

    `EventHandler` objects are typically created by corresponding
    [`EventHandlerPlugin`][ropt.plugins.plan.base.EventHandlerPlugin] factories,
    managed by the [`PluginManager`][ropt.plugins.PluginManager]. Once
    instantiated and added to a `Plan`, their
    [`handle_event`][ropt.plugins.plan.base.EventHandler.handle_event] method is
    called by the plan whenever an [`Event`][ropt.plan.Event] is emitted.

    Handlers can also store state using dictionary-like access (`[]`), allowing
    them to accumulate information or make data available to subsequent steps
    or event handlers within the plan.

    Subclasses must implement the abstract
    [`handle_event`][ropt.plugins.plan.base.EventHandler.handle_event] method to
    define their specific event processing logic.
    """

    def __init__(
        self,
        plan: Plan,
        tags: set[str] | None = None,
        sources: set[PlanComponent | str] | None = None,
    ) -> None:
        """Initialize the EventHandler.

        Associates the event handler with its parent [`Plan`][ropt.plan.Plan],
        assigns a unique ID, and initializes an internal dictionary for storing
        state. The parent plan is accessible via the `plan` property.

        The `sources` parameter acts as a filter, determining which plan steps
        this event handler should listen to. It should be a set containing the
        components or tags that should be handled. When an event is received,
        this event handler checks if the component, or one of its tag, is
        present in the  `sources` set.

        Args:
            plan:    The [`Plan`][ropt.plan.Plan] instance that owns this event handler.
            tags:    Optional tags
            sources: Optional set of steps whose events should be processed.
        """
        super().__init__(plan, tags)
        if sources is None:
            sources = set()
        for source in sources:
            if not isinstance(source, PlanComponent | str):
                msg = "A source must be plan component or a tag string."
                raise TypeError(msg)
        self._sources = (
            {
                source.id if isinstance(source, PlanComponent) else source
                for source in sources
            }
            if sources is not None
            else None
        )
        self.__stored_values: dict[str, Any] = {}

    @property
    def sources(self) -> set[uuid.UUID | str]:
        """Return the source IDs or tags that are listened to.

        Returns:
            The source IDs or tags this event handler is interested in.
        """
        return self._sources

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Process an event emitted by the optimization plan.

        This abstract method must be implemented by concrete `EventHandler`
        subclasses. It defines the event handler's core logic for reacting to
        [`Event`][ropt.plan.Event] objects emitted during the execution of the
        parent [`Plan`][ropt.plan.Plan].

        Implementations should inspect the `event` object (its `event_type` and
        `data`) and perform actions accordingly, such as storing results,
        logging information, or updating internal state.

        Args:
            event: The event object containing details about what occurred in the plan.
        """

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Retrieve a value from the event handler's internal state.

        This method enables dictionary-like access (`handler[key]`) to the
        values stored within the event handler's internal state dictionary. This
        allows handlers to store and retrieve data accumulated during plan
        execution.

        Args:
            key: The string key identifying the value to retrieve.

        Returns:
            The value associated with the specified key.

        Raises:
            AttributeError: If the provided `key` does not exist in the
                            event handler's stored values.
        """
        if key in self.__stored_values:
            return self.__stored_values[key]
        msg = f"Unknown plan variable: `{key}`"
        raise AttributeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Store or update a value in the event handler's internal state.

        This method enables dictionary-like assignment (`handler[key] = value`)
        to store arbitrary data within the event handler's internal state
        dictionary. This allows event handlers to accumulate information or make
        data available to other components of plan.

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


class Evaluator(ABC, PlanComponent):
    """abstract base class for evaluator components within an optimization plan.

    Subclasses must implement the abstract
    [`eval`][ropt.plugins.plan.base.Evaluator.eval] method, which is responsible
    for performing the actual evaluation of variables using an
    [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] and returning an
    [`EvaluatorResult`][ropt.evaluator.EvaluatorResult].
    """

    def __init__(
        self,
        plan: Plan,
        tags: set[str] | None = None,
        clients: set[PlanComponent | str] | None = None,
    ) -> None:
        """Initialize the Evaluator.

        Associates the evaluator with its parent [`Plan`][ropt.plan.Plan], and
        assigns a unique ID. The parent plan is accessible via the `plan`
        property.

        The `clients` parameter acts as a filter, determining which plan steps
        this evaluator should serve. It should be a set containing the the
        components or tags that should be handled. When an evaluation is
        requested, this evaluator checks if the component, or one of its tags,
        is present in the `client` set.

        Args:
            plan:    The [`Plan`][ropt.plan.Plan] instance that owns this evaluator.
            tags:    Optional tags
            clients: The steps that use this evaluator.
        """
        super().__init__(plan, tags)
        if clients is None:
            clients = set()
        for client in clients:
            if not isinstance(client, PlanComponent | str):
                msg = "A client must be plan component or a tag string."
                raise TypeError(msg)
        self.__clients = {
            client.id if isinstance(client, PlanComponent) else client
            for client in clients
        }

    @abstractmethod
    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate objective and constraint functions for given variables.

        This method defines function evaluator callback, which calculates
        objective and constraint functions for a set of variable vectors,
        potentially for a subset of realizations and perturbations.

        Args:
            variables: The matrix of variables to evaluate. Each row represents
                       a variable vector.
            context:   The evaluation context, providing additional information
                       about the evaluation.

        Returns:
            An evaluation results object containing the calculated values.

        Tip: Reusing Objectives and Constraints
            When defining multiple objectives, there may be a need to reuse the
            same objective or constraint value multiple times. For instance, a
            total objective could consist of the mean of the objectives for each
            realization, plus the standard deviation of the same values. This
            can be implemented by defining two objectives: the first calculated
            as the mean of the realizations, and the second using a function
            estimator to compute the standard deviations. The optimizer is
            unaware that both objectives use the same set of realizations. To
            prevent redundant calculations, the evaluator should compute the
            results of the realizations once and return them for both
            objectives.
        """

    @property
    def clients(self) -> set[uuid.UUID | str]:
        """Return the client IDs or tags that are served.

        Returns:
            The IDs of the clients, or the tags, this evaluator will handle.
        """
        return self.__clients

    def add_client(self, client: PlanComponent | str) -> None:
        """Add a client to the evaluator.

        Args:
            client: The client to add.
        """
        if not isinstance(client, PlanComponent | str):
            msg = "A client must be plan component or a tag string."
            raise TypeError(msg)
        id_ = client.id if isinstance(client, PlanComponent) else client
        self.__clients.add(id_)

    def remove_client(self, client: PlanComponent | str) -> None:
        """Remove a client from the evaluator.

        Args:
            client: The client to remove.
        """
        if not isinstance(client, PlanComponent | str):
            msg = "A client must be plan component or a tag string."
            raise TypeError(msg)
        id_ = client.id if isinstance(client, PlanComponent) else client
        if id_ in self.__clients:
            self.__clients.remove(id_)
