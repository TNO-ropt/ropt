"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ropt.exceptions import PlanAborted
from ropt.plugins.plan.base import PlanHandler, PlanStep

if TYPE_CHECKING:
    import uuid

    from ropt.plan import Event

    from ._context import OptimizerContext


class Plan:
    """Plan class for executing optimization workflows.

    The `Plan` object is the core component for executing optimization
    workflows. It orchestrates the execution of individual steps and the
    processing of results through handlers.

    **Building a Plan:**

    To construct a plan, individual actions, known as steps, are added using the
    [`add_step`][ropt.plan.Plan.add_step] method. Data processing and storage
    are managed by handlers, which are added using the
    [`add_handler`][ropt.plan.Plan.add_handler] method. The plan stores the step
    and handler objects internally. Their respective creation functions return
    unique IDs for identification. The
    [`handler_exists`][ropt.plan.Plan.handler_exists] and
    [`step_exists`][ropt.plan.Plan.step_exists] methods can be used to verify
    whether a handler or step with a given name is supported.

    **Executing a Plan:**

    Once a plan is assembled, it can be executed in several ways. For
    fine-grained control, the [`run_step`][ropt.plan.Plan.run_step] method can
    be invoked repeatedly, executing each step individually. This approach
    allows for the integration of complex logic and custom functions, leveraging
    the full capabilities of Python. Alternatively, for more structured
    workflows, a Python function encapsulating a sequence of steps can be
    defined. This function is added to the plan using the
    [`add_function`][ropt.plan.Plan.add_function] method. The entire workflow
    defined by this function can then be executed with a single call to
    [`run_function`][ropt.plan.Plan.run_function], with optional arguments to
    customize its behavior. The [`has_function`][ropt.plan.Plan.has_function]
    method can be used to check if a function has been added to the plan.

    **Shared State and Events:**

    The plan maintains shared state in an
    [`OptimizerContext`][ropt.plan.OptimizerContext], which is provided during
    initialization and can be shared among multiple plans. The
    [`optimizer_context`][ropt.plan.Plan.optimizer_context] property provides
    access to this context.

    Steps can communicate events using the
    [`emit_event`][ropt.plan.Plan.emit_event] method. Result handlers can
    respond to these events, enabling actions such as processing optimization
    results.

    **Nested Plans:**

    Multiple plans can be defined. A step within one plan can trigger the
    execution of another plan, enabling nested workflows. In nested plans, the
    [`set_parent`][ropt.plan.Plan.set_parent] method establishes a parent-child
    relationship, allowing events to propagate up the hierarchy to the parent
    plan.

    **Aborting a Plan:**

    A plan's execution can be terminated, either programmatically from within a
    step or handler, or externally by directly calling the
    [`abort`][ropt.plan.Plan.abort] method. The
    [`aborted`][ropt.plan.Plan.aborted] property can be used to check if a plan
    has been aborted.

    **Handler Data:**

    Individual handlers may store values that they accumulate or calculate from
    the events that they handle. Code outside of the handlers, such as the
    optimization workflow code that runs the steps, can set and retrieve these
    values using the [`get`][ropt.plan.Plan.get] and [`set`][ropt.plan.Plan.set]
    methods.
    """

    def __init__(
        self,
        optimizer_context: OptimizerContext,
        parent: Plan | None = None,
    ) -> None:
        """Initialize a plan object.

        Constructs a new plan, associating it with an
        [`OptimizerContext`][ropt.plan.OptimizerContext] and an optional parent
        plan.

        The plan will operate within the provided `optimizer_context`. If a
        `parent` plan is specified, this plan becomes a child, enabling event
        propagation up the plan hierarchy.

        Args:
            optimizer_context: The execution context for the plan.
            parent: An optional parent plan.
        """
        self._optimizer_context = optimizer_context

        self._aborted = False
        self._parent = parent
        self._handlers: dict[uuid.UUID, PlanHandler] = {}
        self._steps: dict[uuid.UUID, PlanStep] = {}
        self._function: Callable[..., Any] | None = None

    @property
    def aborted(self) -> bool:
        """Check if the plan was aborted.

        Determines whether the plan's execution has been aborted.

        Returns:
            bool: `True` if the plan was aborted; otherwise, `False`.
        """
        return self._aborted

    @property
    def optimizer_context(self) -> OptimizerContext:
        """Return the optimizer context.

        Retrieves the [`OptimizerContext`][ropt.plan.OptimizerContext] object
        associated with this plan. The optimizer context provides shared state
        and functionality for executing the optimization plan.

        Returns:
            OptimizerContext: The optimizer context object used by the plan.
        """
        return self._optimizer_context

    def add_handler(self, name: str, **kwargs: Any) -> uuid.UUID:  # noqa: ANN401
        """Add a handler to the plan.

        Constructs and registers a result handler with the plan. The handler's
        type is determined by the provided `name`, which the plugin system uses
        to locate the corresponding handler class. Any additional keyword
        arguments are passed to the handler's constructor.

        Args:
            name:   The name of the handler to add.
            kwargs: Additional arguments for the handler's constructor.

        Returns:
            The unique ID of the newly added handler.
        """
        handler = self._optimizer_context.plugin_manager.get_plugin(
            "plan_handler", method=name
        ).create(name, self, **kwargs)
        assert isinstance(handler, PlanHandler)
        self._handlers[handler.id] = handler
        return handler.id

    def add_step(self, name: str, **kwargs: Any) -> uuid.UUID:  # noqa: ANN401
        """Add a step to the plan.

        Registers a step with the plan. The step's type is determined by the
        provided `name`, which the plugin system uses to locate the
        corresponding step class. Any additional keyword arguments are passed to
        the step's constructor.

        Args:
            name:   The name of the step to add.
            kwargs: Additional arguments for the step's constructor.

        Returns:
            uuid.UUID: The unique ID of the newly added step.
        """
        step = self._optimizer_context.plugin_manager.get_plugin(
            "plan_step", method=name
        ).create(name, self, **kwargs)
        assert isinstance(step, PlanStep)
        self._steps[step.id] = step
        return step.id

    def handler_exists(self, name: str) -> bool:
        """Check if a handler exists.

        Determines whether a handler with the specified name is supported by the
        plugin system.

        Args:
            name: The name of the handler to check.

        Returns:
            bool: `True` if the handler exists; otherwise, `False`.
        """
        return self._optimizer_context.plugin_manager.is_supported(
            "plan_handler", method=name
        )

    def step_exists(self, name: str) -> bool:
        """Check if a step exists.

        Determines whether a step with the specified name is supported by the
        plugin system.

        Args:
            name: The name of the step to check.

        Returns:
            bool: `True` if the step exists; otherwise, `False`.
        """
        return self._optimizer_context.plugin_manager.is_supported(
            "plan_step", method=name
        )

    def run_step(self, step: uuid.UUID, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run a step in the plan.

        Executes a specific step within the plan. The step's `run` method is
        called with the provided keyword arguments. If the plan has been
        aborted, a [`PlanAborted`][ropt.exceptions.PlanAborted] exception is
        raised before the step is executed.

        The step is executed only once. The value returned by the step's `run`
        method is returned by this method.

        Args:
            step:   The unique ID of the step to run.
            kwargs: Additional arguments to pass to the step's `run` method.

        Returns:
            Any: The value returned by the step's `run` method.

        Raises:
            AttributeError: If the provided `step` ID is not valid.
            PlanAborted:    If the plan has been aborted.
        """
        if step not in self._steps:
            msg = "not a valid step"
            raise AttributeError(msg)
        if self._aborted:
            msg = "Plan was aborted by the previous step."
            raise PlanAborted(msg)
        return self._steps[step].run(**kwargs)

    def add_function(self, func: Callable[..., Any]) -> None:
        """Add a function to the plan.

        Registers a user-defined function with the plan. This function can
        encapsulate a sequence of steps or custom logic. It can be executed
        later using the [`run_function`][ropt.plan.Plan.run_function] method.

        Args:
            func: The function to register with the plan.
        """
        self._function = func

    def has_function(self) -> bool:
        """Check if a function has been added to the plan.

        Determines whether a user-defined function has been registered with the
        plan.

        Returns:
            bool: `True` if a function has been added; otherwise, `False`.
        """
        return self._function is not None

    def run_function(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run a function in the plan.

        Executes the user-defined function that has been registered with the
        plan via the [`add_function`][ropt.plan.Plan.add_function] method.

        Args:
            args:   Arbitrary positional arguments to pass to the function.
            kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            Any: The result returned by the function.

        Raises:
            AttributeError: If no function has been added to the plan.
        """
        if self._function is None:
            msg = "No function has been added to the plan."
            raise AttributeError(msg)
        return self._function(self, *args, **kwargs)

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
        plan. This enables event propagation up the plan hierarchy.

        Args:
            parent: The parent plan.
        """
        self._parent = parent

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Emits an event, triggering associated handlers and observers.

        When this method is called:

        1.  All event handlers associated with the plan are invoked.
        2.  If the plan has no parent, all observer functions registered for the
            specified event type are called.
        3.  If the plan has a parent, the parent plan's `emit_event` method is
            also called, propagating the event up the hierarchy.

        Args:
            event: The event object to emit.
        """
        for handler in self._handlers.values():
            handler.handle_event(event)
        if self._parent is None:
            self._optimizer_context.call_observers(event)
        else:
            self._parent.emit_event(event)

    def get(self, id_: uuid.UUID, /, key: str) -> Any:  # noqa: ANN401
        """Retrieve a value stored in a handler.

        Retrieves a value stored within a specific result handler. This method
        uses the `[]` operator to access the value associated with the given
        key.

        Args:
            id_: The unique identifier of the handler.
            key: The key associated with the value to retrieve.

        Returns:
            Any: The value associated with the key in the specified handler.

        Raises:
            AttributeError: If the provided `id_` is not a valid handler ID.
        """
        if id_ not in self._handlers:
            msg = "not a valid handler"
            raise AttributeError(msg)
        return self._handlers[id_][key]

    def set(self, id_: uuid.UUID, /, key: str, value: Any) -> None:  # noqa: ANN401
        """Set a value in a handler.

        Stores a value within a specific result handler. This method uses the
        `[]` operator to assign the value to the specified key.

        Args:
            id_:   The unique identifier of the handler.
            key:   The key to associate with the value.
            value: The value to store.

        Raises:
            AttributeError: If the provided `id_` is not a valid handler ID.
        """
        if id_ not in self._handlers:
            msg = "not a valid handler"
            raise AttributeError(msg)
        self._handlers[id_][key] = value
