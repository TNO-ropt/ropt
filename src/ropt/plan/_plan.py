"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ropt.plugins.plan.base import Evaluator, EventHandler, PlanComponent, PlanStep

if TYPE_CHECKING:
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
    [`event_handlers`][ropt.plan.Plan.event_handlers] property. Event handlers
    can respond to these events, enabling actions such as processing
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
    execution of another plan, enabling nested workflows.

    **Handler Data:**

    Individual event handlers may store values that they accumulate or calculate
    from the events that they handle. Code outside of the event handlers, such
    as the optimization workflow code that runs the steps, can set and retrieve
    these values using the `[]` operator.
    """

    def __init__(
        self,
        plugin_manager: PluginManager,
        *,
        event_handlers: list[EventHandler] | None = None,
        evaluators: list[Evaluator] | None = None,
    ) -> None:
        """Initialize a plan object.

        Constructs a new plan, associating it with an evaluator, and optionally
        with a plugin manager and/or a event handlers and evaluators.

        The `plugin_manager` is used by the plan, and possibly by steps and
        event handlers to add plugin functionality.

        Args:
            plugin_manager: A plugin manager.
            event_handlers: Event handlers to add to the plan.
            evaluators:     Evaluators to add to the plan.
        """
        self._plugin_manager = plugin_manager
        self._handlers: list[EventHandler] = []
        self._parent_handlers: list[EventHandler] = (
            [] if event_handlers is None else event_handlers
        )
        self._evaluators: list[Evaluator] = []
        self._parent_evaluators: list[Evaluator] = (
            [] if evaluators is None else evaluators
        )

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
        *,
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
            name, tags=tags, sources=sources, **kwargs
        )
        assert isinstance(handler, EventHandler)
        self._handlers.append(handler)
        return handler

    def add_step(
        self,
        name: str,
        *,
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
            name, self, tags=tags, **kwargs
        )
        assert isinstance(step, PlanStep)
        return step

    def add_evaluator(
        self,
        name: str,
        *,
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
            name, tags=tags, clients=clients, **kwargs
        )
        assert isinstance(evaluator, Evaluator)
        self._evaluators.append(evaluator)
        return evaluator

    @property
    def event_handlers(self) -> list[EventHandler]:
        """Get the event handlers available to this plan.

        Returns:
            A list of handlers.
        """
        return self._handlers + self._parent_handlers

    @property
    def evaluators(self) -> list[Evaluator]:
        """Get the evaluators available to this plan.

        Returns:
            A list of evaluators instances.
        """
        return self._evaluators + self._parent_evaluators
