"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ropt.plugins.plan.base import PlanStep

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
    optimization results. Event handlers are added to a step using the
    [`add_event_handler`][ropt.plan.Plan.add_event_handler] method.

    **Evaluators:**

    Evaluators ([`Evaluator`][ropt.plugins.plan.base.Evaluator]) are key
    components responsible for performing function evaluations, such as
    computing objective functions or constraint values. They are added to the
    step using the [`add_evaluator`][ropt.plan.Plan.add_evaluator] method.

    **Nested Plans:**

    Multiple plans can be defined. A step within one plan can trigger the
    execution of another plan, enabling nested workflows.

    **Handler Data:**

    Individual event handlers may store values that they accumulate or calculate
    from the events that they handle. Code outside of the event handlers, such
    as the optimization workflow code that runs the steps, can set and retrieve
    these values using the `[]` operator.
    """

    def __init__(self, plugin_manager: PluginManager) -> None:
        """Initialize a plan object.

        Constructs a new plan, associating it with  a plugin manager and event handlers.

        The `plugin_manager` is used by the plan, and possibly by steps and
        event handlers to add plugin functionality.

        Args:
            plugin_manager: A plugin manager.
            event_handlers: Event handlers to add to the plan.
        """
        self._plugin_manager = plugin_manager

    @property
    def plugin_manager(self) -> PluginManager:
        """Return the plugin manager.

        Retrieves the [`PluginManager`][ropt.plugins.PluginManager] object
        associated with this plan.

        Returns:
            The plugin manager stored by the plan.
        """
        return self._plugin_manager

    def add_step(
        self,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> PlanStep:
        """Add a step to the plan.

        Registers a step with the plan. The step's type is determined by the
        provided `name`, which the plugin system uses to locate the
        corresponding step class. Any additional keyword arguments are passed to
        the step's constructor.

        Args:
            name:   The name of the step to add.
            kwargs: Additional arguments for the step's constructor.

        Returns:
            The newly added step.
        """
        step = self._plugin_manager.get_plugin("plan_step", method=name).create(
            name, self, **kwargs
        )
        assert isinstance(step, PlanStep)
        return step
