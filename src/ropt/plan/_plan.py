"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ropt.exceptions import PlanAborted
from ropt.plugins.plan.base import Evaluator, EventHandler, PlanComponent, PlanStep

if TYPE_CHECKING:
    import uuid

    from ropt.enums import EventType
    from ropt.plan import Event
    from ropt.plugins import PluginManager


class Plan:
    """Plan class for executing optimization workflows.

    The `Plan` object is the core component for executing optimization workflows.
    It orchestrates the execution of individual steps, manages evaluators for
    function computations, and processes data and results through event handlers.

    **Building a Plan:**

    A `Plan` is constructed by adding three main types of components, typically
    instantiated via the [`PluginManager`][ropt.plugins.PluginManager]:

    1.  **Steps ([`PlanStep`][ropt.plugins.plan.base.PlanStep]):** These define
        individual actions or operations within the optimization workflow. Steps
        are added using the [`add_step`][ropt.plan.Plan.add_step] method.
    2.  **Event Handlers ([`EventHandler`][ropt.plugins.plan.base.EventHandler]):**
        These components process data, track results, or react to events
        emitted during plan execution. Event handlers are added using the
        [`add_event_handler`][ropt.plan.Plan.add_event_handler] method.
    3.  **Evaluators ([`Evaluator`][ropt.plugins.plan.base.Evaluator]):** These
        are responsible for performing function evaluations (e.g., objective
        functions, constraints). Evaluators are added using the
        [`add_evaluator`][ropt.plan.Plan.add_evaluator] method and can be
        passed to steps that need them.

    **Tags:**

    Steps, event handlers, and evaluators can be assigned one or more tags.
    These tags can be used to identify the components instead of their unique
    IDs. Unlike ID's, tags do not need to be unique. This is useful when the
    components are created dynamically or if multiple components are to be
    identified as a group. For example, when specifying the source of events
    that a handler should process, its `sources` argument may contain both
    component objects, which identifies by their ID, or tags, which could refer
    to multiple components that have that tag.

    **Executing a Plan:**

    Once a plan is assembled, the [`run`][ropt.plugins.plan.base.PlanStep.run]
    method can be invoked for each step individually. This approach allows for
    the integration of complex logic and custom functions, leveraging the full
    capabilities of Python.

    **PluginManager:**

    A [`PluginManager`][ropt.plugins.PluginManager] object can be provided that
    is used by the plan object to find and instantiate step, event handler, and
    evaluator objects. This manager can also be used by these components to
    implement further plugin-based functionality.

    **Events:**

    Steps can communicate events by retrieving a list of handlers using the
    [`get_event_handlers`][ropt.plan.Plan.get_event_handlers] method. Event
    handlers can respond to these events, enabling actions such as processing
    optimization results. Event handlers are added to the plan using the
    [`add_event_handler`][ropt.plan.Plan.add_event_handler] method. To connect
    the event handlers to steps, they generally accept a set of steps via the
    `sources` argument. The steps must be part of the same plan, or a child plan
    (if existent).

    **Evaluators:**

    Evaluators ([`Evaluator`][ropt.plugins.plan.base.Evaluator]) are key
    components responsible for performing function evaluations, such as
    computing objective functions or constraint values. They are added to the
    plan using the [`add_evaluator`][ropt.plan.Plan.add_evaluator] method. They
    connect to the steps in the plan, or in child plans, via the `clients`
    argument.

    **Nested Plans:**

    Multiple plans can be defined. A step within one plan can trigger the
    execution of another plan, enabling nested workflows. In nested plans, the
    [`set_parent`][ropt.plan.Plan.set_parent] method establishes a parent-child
    relationship. This allows events to propagate up the hierarchy to the parent
    plan.

    **Aborting a Plan:**

    A plan's execution can be terminated, either programmatically from within a
    step or event handler, or externally by directly calling the
    [`abort`][ropt.plan.Plan.abort] method. The
    [`aborted`][ropt.plan.Plan.aborted] property can be used to check if a plan
    has been aborted.

    **Handler Data:**

    Individual event handlers may store values that they accumulate or calculate
    from the events that they handle. Code outside of the event handlers, such
    as the optimization workflow code that runs the steps, can set and retrieve
    these values using the `[]` operator.
    """

    def __init__(
        self,
        plugin_manager: PluginManager,
        parent: Plan | None = None,
    ) -> None:
        """Initialize a plan object.

        Constructs a new plan, associating it with an evaluator, and optionally
        with a plugin manager and/or a parent plan.

        The `plugin_manager` is used by the plan, and possibly by steps and
        event handlers to add plugin functionality.

        If a `parent` plan is specified, this plan becomes a child, enabling
        communication up the plan hierarchy.

        Args:
            plugin_manager: A plugin manager.
            parent:         An optional parent plan.
        """
        self._plugin_manager = plugin_manager
        self._aborted = False
        self._parent = parent
        self._handlers: dict[uuid.UUID, EventHandler] = {}
        self._steps: dict[uuid.UUID, PlanStep] = {}
        self._evaluators: dict[uuid.UUID, Evaluator] = {}

    @property
    def aborted(self) -> bool:
        """Check if the plan was aborted.

        Determines whether the plan's execution has been aborted.

        Returns:
            bool: `True` if the plan was aborted; otherwise, `False`.
        """
        return self._aborted

    @property
    def plugin_manager(self) -> PluginManager:
        """Return the plugin manager.

        Retrieves the [`PluginManager`][ropt.plugins.PluginManager] object
        associated with this plan.

        Returns:
            The plugin manager stored by the plan.
        """
        return self._plugin_manager

    def add_event_handler(
        self,
        name: str,
        tags: set[str] | None = None,
        sources: set[PlanComponent | str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> EventHandler:
        """Add an event handler to the plan.

        Constructs and registers an event handler with the plan. The handler's
        type is determined by the provided `name`, which the plugin system uses
        to locate the corresponding handler class. Any additional keyword
        arguments are passed to the handler's constructor.

        The `sources` parameter acts as a filter, determining which plan steps
        this event handler should listen to. It should be a set containing the
        `PlanStep` instances whose event you want to receive. When an event is
        received, this event handler checks if the step that emitted the event
        (`event.source`) is present in the `sources` set. If `sources` is
        `None`, events from all sources will be processed.

        Args:
            name:    The name of the event handler to add.
            tags:    Optional tags
            sources: The steps whose events should be processed.
            kwargs:  Additional arguments for the handler's constructor.

        Returns:
            The newly added event handler.
        """
        handler = self._plugin_manager.get_plugin("event_handler", method=name).create(
            name, self, tags, sources, **kwargs
        )
        assert isinstance(handler, EventHandler)
        self._handlers[handler.id] = handler
        return handler

    def add_step(
        self,
        name: str,
        tags: set[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> PlanStep:
        """Add a step to the plan.

        Registers a step with the plan. The step's type is determined by the
        provided `name`, which the plugin system uses to locate the
        corresponding step class. Any additional keyword arguments are passed to
        the step's constructor.

        Args:
            name:   The name of the step to add.
            tags:   Optional tags
            kwargs: Additional arguments for the step's constructor.

        Returns:
            The newly added step.
        """
        step = self._plugin_manager.get_plugin("plan_step", method=name).create(
            name, self, tags, **kwargs
        )
        assert isinstance(step, PlanStep)
        self._steps[step.id] = step
        return step

    def add_evaluator(
        self,
        name: str,
        tags: set[str] | None = None,
        clients: set[PlanComponent | str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Evaluator:
        """Add an evaluator object to the plan.

        Creates an evaluator of a type that is determined by the provided `name`,
        which the plugin system uses to locate the corresponding evaluator class.
        Any additional keyword arguments are passed to the evaluators's constructor.

        The `clients` parameter acts as a filter, determining which plan steps
        this evaluator should serve. It should be a set containing the
        `PlanStep` instances that should be handled. When an evaluation is
        requested, this evaluator checks if the step is present in the `client`
        set.

        Args:
            name:    The name of the evaluator to add.
            tags:    Optional tags
            clients: The clients that should be served by this evaluator.
            kwargs:  Additional arguments for the evaluators's constructor.

        Returns:
            The new evaluator object.
        """
        evaluator = self._plugin_manager.get_plugin("evaluator", method=name).create(
            name, self, tags, clients, **kwargs
        )
        assert isinstance(evaluator, Evaluator)
        self._evaluators[evaluator.id] = evaluator
        return evaluator

    def run_step(self, step: PlanStep, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run a step in the plan.

        Executes a specific step within the plan. The step's `run` method is
        called with the provided keyword arguments. If the plan has been
        aborted, a [`PlanAborted`][ropt.exceptions.PlanAborted] exception is
        raised before the step is executed.

        The step is executed only once. The value returned by the step's `run`
        method is returned by this method.

        Args:
            step:   Tthe step to run.
            kwargs: Additional arguments to pass to the step's `run` method.

        Returns:
            Any: The value returned by the step's `run` method.

        Raises:
            AttributeError: If the provided `step` ID is not valid.
            PlanAborted:    If the plan has been aborted.
        """
        if step.id not in self._steps:
            msg = "not a valid step"
            raise AttributeError(msg)
        if self._aborted:
            msg = "Plan was aborted by the previous step."
            raise PlanAborted(msg)
        return step.run_step_from_plan(**kwargs)

    def get_event_handlers(
        self, source: PlanComponent, event_types: set[EventType]
    ) -> dict[EventType, list[Callable[[Event], None]]]:
        """Get the event handlers for a given source and event types.

        When this method is called, all event handlers associated with the plan
        are searched for those that handle the `source`. Then, if the plan has a
        parent, the parent plan's `get_event_handlers` method is also called,
        find handlers further up the hierarchy.

        Args:
            source:      The source of the event.
            event_types: The event types that should be handled.

        Returns:
            A mapping of event types to a list of suitable handlers.
        """
        result: dict[EventType, list[Callable[[Event], None]]] = {}
        for handler in self._handlers.values():
            if (source.id in handler.sources or source.tags & handler.sources) and (
                event_types & handler.event_types
            ):
                for event_type in handler.event_types:
                    if event_type in result:
                        result[event_type].append(handler.handle_event)
                    else:
                        result[event_type] = [handler.handle_event]
        if self._parent is not None:
            parent_handlers = self._parent.get_event_handlers(source, event_types)
            for event_type, handlers in parent_handlers.items():
                if event_type not in result:
                    result[event_type] = handlers
                else:
                    result[event_type].extend(handlers)
        return result

    def _get_evaluators(self, client: PlanComponent) -> list[Evaluator]:
        evaluators = [
            evaluator
            for evaluator in self._evaluators.values()
            if client.id in evaluator.clients or client.tags & evaluator.clients
        ]
        if not evaluators and self._parent is not None:
            evaluators = self._parent._get_evaluators(client)  # noqa: SLF001
        return evaluators

    def get_evaluator(self, client: PlanComponent) -> Evaluator:
        """Retrieve the appropriate evaluator for a given client step.

        This method searches for an [`Evaluator`][ropt.plugins.plan.base.Evaluator]
        that is configured to serve the specified `client`
        ([`PlanStep`][ropt.plugins.plan.base.PlanStep]). The search starts in the
        current plan and, if no suitable evaluator is found and a parent plan
        exists, continues recursively up the plan hierarchy.

        An evaluator is considered suitable the `id` or `tags` of the `client`
        step is present in the `clients` set of the evaluator.

        The method expects to find exactly one suitable evaluator.

        Args:
            client: The step for which an evaluator is being requested.

        Returns:
            The single evaluator instance configured to serve the client.

        Raises:
            AttributeError: If no suitable evaluator is found, or if multiple
                            suitable evaluators are found.
        """
        evaluators = self._get_evaluators(client)
        if not evaluators:
            msg = "No suitable evaluator found."
            raise AttributeError(msg)
        if len(evaluators) > 1:
            msg = "Ambiguous request: multiple suitable evaluators found."
            raise AttributeError(msg)
        return evaluators[0]

    def abort(self) -> None:
        """Abort the plan.

        Prevents further steps in the plan from being executed. This method does
        not interrupt a currently running step but ensures that no subsequent
        steps will be initiated. It can be used to halt the plan's execution due
        to a step failure or external intervention.

        The [`aborted`][ropt.plan.Plan.aborted] property can be used to check if
        the plan has been aborted.
        """
        self._aborted = True

    def set_parent(self, parent: Plan) -> None:
        """Set the parent of the plan.

        Establishes a parent-child relationship between this plan and another
        plan. This enables event propagation up the plan hierarchy. It also
        allows the `get_evaluation` method to inquire the parent for an
        evaluator object, if necessary.

        Args:
            parent: The parent plan.
        """
        self._parent = parent
