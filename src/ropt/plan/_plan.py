"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ropt.exceptions import PlanAborted
from ropt.plugins.plan.base import PlanStep, ResultHandler

if TYPE_CHECKING:
    from ropt.plan import Event

    from ._context import OptimizerContext


class Plan:
    """The plan class for executing optimization workflows."""

    def __init__(
        self,
        optimizer_context: OptimizerContext,
        parent: Plan | None = None,
    ) -> None:
        """Initialize a plan object.

        This method initializes a plan using an
        [`OptimizerContext`][ropt.plan.OptimizerContext] object. An optional
        [`plugin_manager`][ropt.plugins.PluginManager] argument allows for the
        specification of custom plugins for result handlers and step objects
        within the plan. If omitted, only plugins installed through Python's
        standard entry points are used.

        Args:
            optimizer_context: The context in which the plan will execute,
                               providing shared resources across steps.
            parent:            Optional reference to a parent plan.
        """
        self._optimizer_context = optimizer_context

        self._aborted = False
        self._parent = parent
        self._handlers: list[ResultHandler] = []
        self._function: Callable[..., Any] | None = None

    def add_handler(self, name: str, **kwargs: Any) -> ResultHandler:  # noqa: ANN401
        """Add a handler to the plan.

        Args:
            name: The name of the handler to add.
            kwargs: Additional arguments to pass to the handler.

        Returns:
            The new handler.
        """
        handler = self._optimizer_context.plugin_manager.get_plugin(
            "plan_handler", method=name
        ).create(name, self, **kwargs)
        self._handlers.append(handler)
        assert isinstance(handler, ResultHandler)
        return handler

    def handler_exists(self, name: str) -> bool:
        """Check if a handler exists.

        Args:
            name: The name of the handler to check.

        Returns:
            Whether the handler exists.
        """
        return self._optimizer_context.plugin_manager.is_supported(
            "plan_handler", method=name
        )

    def clear_handlers(self) -> None:
        """Clear all handlers from the plan."""
        self._handlers.clear()

    def add_step(self, name: str, **kwargs: Any) -> PlanStep:  # noqa: ANN401
        """Add a step to the plan.

        Args:
            name: The name of the step to run.
            kwargs: Additional arguments to pass to the step.

        Returns:
            The step that was executed.
        """
        step = self._optimizer_context.plugin_manager.get_plugin(
            "plan_step", method=name
        ).create(name, self, **kwargs)
        assert isinstance(step, PlanStep)
        return step

    def step_exists(self, name: str) -> bool:
        """Check if a step exists.

        Args:
            name: The name of the step to check.

        Returns:
            Whether the step exists.
        """
        return self._optimizer_context.plugin_manager.is_supported(
            "plan_step", method=name
        )

    def add_function(self, func: Callable[..., Any]) -> None:
        """Add a function to the plan.

        The function can be called using the `run_function` method, and is
        generally used to execute one or more steps in the plan.

        Args:
            func: The function to add to the plan.
        """
        self._function = func

    def run_step(self, step: PlanStep, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run a step in the plan.

        Args:
            step:   The step to run
            kwargs: Additional arguments to pass to the step.

        Raises:
            PlanAborted: _description_
        """
        if self._aborted:
            msg = "Plan was aborted by the previous step."
            raise PlanAborted(msg)
        return step.run(**kwargs)

    def run_function(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run a function in the plan.

        Args:
            args:   Arbitrary positional arguments to pass to the function.
            kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            The result of the function.
        """
        if self._function is None:
            msg = "No function has been added to the plan."
            raise AttributeError(msg)
        return self._function(self, *args, **kwargs)

    def abort(self) -> None:
        """Abort the plan.

        This will not abort a running plan, but will prevent any further steps
        from being executed. This can be used to stop the plan from continuing
        if a step fails or if the user decides to stop the optimization.

        The `aborted` method can be used to check if the plan was aborted.
        """
        self._aborted = True

    @property
    def aborted(self) -> bool:
        """Check if the plan was aborted.

        Returns:
            bool: `True` if the plan was aborted; otherwise, `False`.
        """
        return self._aborted

    @property
    def optimizer_context(self) -> OptimizerContext:
        """Return the optimizer context associated with the plan.

        This method retrieves the optimizer context object that provides shared
        state and functionality for executing the optimization plan.

        Returns:
            The optimizer context object used by the plan.
        """
        return self._optimizer_context

    def set_parent(self, parent: Plan) -> None:
        self._parent = parent

    def emit_event(self, event: Event) -> None:
        """Emit an event of the specified type with the provided data.

        When this method is called, the following steps are executed:

        1. All event handlers associated with the plan are invoked.
        2. If the plan has no parent, all observer functions registered for the
           specified event type via the `add_observer` method are called.
        3. If the plan has a parent, the `emit_event` method of the parent plan
           is also called.

        Args:
            event: The event object to emit.
        """
        for handler in self._handlers:
            handler.handle_event(event)
        if self._parent is None:
            self._optimizer_context.call_observers(event)
        else:
            self._parent.emit_event(event)
