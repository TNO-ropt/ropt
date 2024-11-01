"""This module defines the optimization plan object."""

from __future__ import annotations

import sys
from itertools import chain, count
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from numpy.random import default_rng

from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.enums import EventType
from ropt.plugins import PluginManager

from ._expr import ExpressionEvaluator

if TYPE_CHECKING:
    from ropt.config.plan import PlanConfig, PlanStepConfig
    from ropt.evaluator import Evaluator
    from ropt.plan import Event
    from ropt.plugins.plan.base import PlanStep, ResultHandler

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class OptimizerContext:
    """Context class for shared state across a plan.

    An optimizer context object holds the information and state shared across
    all steps in an optimization plan. This currently includes the following:

    - An [`Evaluator`][ropt.evaluator.Evaluator] callable for evaluating
      functions.
    - A seed for a random number generator used in stochastic gradient
      estimation.
    - An expression evaluator object.
    - An iterator that generates unique result IDs.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        seed: Optional[int] = None,
        expr: Optional[ExpressionEvaluator] = None,
    ) -> None:
        """Initialize the optimization context.

        Args:
            evaluator: The callable for running function evaluations
            seed:      Optional seed for the random number generator
            expr:      Optional expression evaluator
        """
        self.evaluator = evaluator
        self.rng = default_rng(DEFAULT_SEED) if seed is None else default_rng(seed)
        self.expr = ExpressionEvaluator() if expr is None else expr
        self.result_id_iter = count()
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            event: [] for event in EventType
        }

    def add_observer(
        self,
        event: EventType,
        callback: Callable[[Event], None],
    ) -> Self:
        """Add an observer function.

        Observer functions are called during optimization when an event of the
        specified type occurs. The provided callable must accept a single
        argument of the [`Event`][ropt.plan.Event] class, which contains
        information about the occurred event.

        Note:
            Before the observer functions are called, all result handlers are
            executed, which may potentially modify the event.

        Args:
            event:    The type of events that the observer will react to.
            callback: The function to call when the specified event is received.
                      This function should accept one argument, which will be
                      an instance of the [`Event`][ropt.plan.Event] class.

        Returns:
            Self, to allow for method chaining.
        """
        self._subscribers[event].append(callback)
        return self

    def call_observers(self, event: Event) -> None:
        """Call observers for a specified event.

        This method invokes all observers associated with the context for the
        type of event passed as an argument.

        Args:
            event: The event that is emitted.
        """
        for callback in self._subscribers[event.event_type]:
            callback(event)


class Plan:
    """The plan class for executing optimization workflows."""

    def __init__(
        self,
        config: PlanConfig,
        optimizer_context: OptimizerContext,
        plugin_manager: Optional[PluginManager] = None,
        parent: Optional[Plan] = None,
        plan_id: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize a plan object.

        This method initializes a plan using a `PlanConfig` object and an
        `OptimizationContext` object. An optional `plugin_manager` argument
        allows the specification of custom plugins for result handlers and step
        objects within the plan. If omitted, only plugins installed through
        Python's standard entry points are used.

        Plans can spawn additional plans within their workflow, and any spawned
        plan receives a reference to the original plan via the `parent`
        argument.

        Each plan is assigned a unique integer sequence number at creation,
        which is stored as a tuple of sequence numbers reflecting plan nesting.
        When spawning a child plan with the [`spawn`][ropt.plan.Plan.spawn]
        method, the new plan receives an incremented sequence number at the
        beginning. This creates an ordered sequence of IDs that uniquely
        reflects the hierarchy and order of spawned plans.

        Args:
            config:            The configuration for the optimizer.
            optimizer_context: The context in which the plan will execute,
                               providing shared resources across steps.
            plugin_manager:    Optional custom plugin manager for step and
                               result handler plugins.
            parent:            Optional reference to the parent plan that
                               spawned this plan.
            plan_id:           A list of plan IDs reflecting the plan hierarchy.
        """
        self._plan_config = config
        self._optimizer_context = optimizer_context
        self._vars: Dict[str, Any] = {}
        self._plan_ids: Tuple[int, ...] = (0,) if plan_id is None else plan_id
        self._plan_ids = (*self._plan_ids, -1)

        self._plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )
        for var in chain(
            self._plan_config.inputs,
            self._plan_config.outputs,
            self._plan_config.variables,
        ):
            if var in self._vars:
                msg = f"Plan variable already exists: `{var}`"
                raise AttributeError(msg)
            if var == "plan_id":
                msg = f"Plan variable overrides a builtin variable: {var}"
                raise AttributeError(msg)
            self._set_item(var, None)
        self._set_item("plan_id", list(self.plan_id))
        self._steps = self.create_steps(config.steps)
        self._handlers: List[ResultHandler] = [
            self._plugin_manager.get_plugin("plan", method=config.run).create(
                config, self
            )
            for config in config.results
        ]
        self._aborted = False
        self._parent = parent

    def run(self, *args: Any) -> Tuple[Any, ...]:  # noqa: ANN401
        """Run the Plan.

        This method accepts an arbitrary number of inputs that are stored in the
        plan variables that are specified in the `inputs` section of the plan
        [`configuration`][ropt.config.plan.PlanConfig]. The number of arguments
        must be equal to the number elements in that section otherwise an error
        will be raised.

        After execution of the steps, this method will return the contents of
        the plan variables that are specified in the `outputs` section of the
        plan [`configuration`][ropt.config.plan.PlanConfig] as a tuple.
        """
        for var, value in self._plan_config.variables.items():
            self[var] = self.eval(value)
        len_args = len(args)
        len_inputs = len(self._plan_config.inputs)
        if len_args != len_inputs:
            msg = f"The number of inputs is incorrect: expected {len_inputs}, passed {len_args}"
            raise RuntimeError(msg)
        for name, arg in zip(self._plan_config.inputs, args):
            self[name] = arg
        self.run_steps(self._steps)
        missing = [name for name in self._plan_config.outputs if name not in self]
        if missing:
            msg = f"Missing outputs: {missing}"
            raise RuntimeError(msg)
        return tuple(self[name] for name in self._plan_config.outputs)

    def abort(self) -> None:
        """Abort the plan."""
        self._aborted = True

    @property
    def plan_id(self) -> Tuple[int, ...]:
        """Return the list of plan IDs.

        Plans can spawn other plans sequentially, in parallel, or as nested
        workflows. The plan IDs uniquely represent the order and nesting of
        these plans, using a list of sequence numbers.

        Returns:
            The list of plan IDs for this plan.
        """
        return self._plan_ids[:-1]

    @property
    def aborted(self) -> bool:
        """Check if the plan was aborted by the user.

        Returns:
            bool: `True` if the plan was aborted by the user; otherwise, `False`.
        """
        return self._aborted

    def create_steps(self, step_configs: List[PlanStepConfig]) -> List[PlanStep]:
        """Instantiate step objects from step configurations.

        This method takes a list of step configuration objects and creates a
        corresponding list of initialized step objects, each configured based on
        its respective configuration.

        Args:
            step_configs: List of configuration objects defining each step.

        Returns:
            List of configured step objects ready for execution in the plan.
        """
        return [
            self._plugin_manager.get_plugin("plan", method=step_config.run).create(
                step_config, self
            )
            for step_config in step_configs
        ]

    def run_steps(self, steps: List[PlanStep]) -> None:
        """Execute a list of steps in the plan.

        This method iterates through and executes a provided list of plan steps.
        If execution is interrupted by the user, it returns `True` to indicate
        an aborted run.

        Args:
            steps: A list of steps to be executed sequentially.

        Returns:
            `True` if execution was aborted by the user; otherwise, `False`.
        """
        for task in steps:
            if self._check_condition(task.step_config):
                task.run()
            if self._aborted:
                break

    @property
    def plugin_manager(self) -> PluginManager:
        """Return the plugin manager associated with the plan.

        This method retrieves the plugin manager that is used to manage plugins
        for the plan's steps and result handlers.

        Returns:
            The plugin manager instance used by the plan.
        """
        return self._plugin_manager

    @property
    def optimizer_context(self) -> OptimizerContext:
        """Return the optimizer context associated with the plan.

        This method retrieves the optimizer context object that provides shared
        state and functionality for executing the optimization plan.

        Returns:
            The optimizer context object used by the plan.
        """
        return self._optimizer_context

    def spawn(self, config: PlanConfig) -> Plan:
        """Spawn a new plan from the current plan.

        This method creates a new plan that shares the same optimization context
        and plugin manager as the current plan. However, it does not inherit
        other properties, such as variables.

        Any signals emitted within the spawned plan are forwarded to the result
        handlers of the current plan and to the connected observers. In other
        words, signals emitted by the spawned plan will "bubble up" to the
        current plan.

        Args:
            config: The configuration for the new plan.

        Returns:
            A new plan object configured with the provided configuration.
        """
        self._plan_ids = (*self._plan_ids[:-1], self._plan_ids[-1] + 1)
        return Plan(
            config,
            optimizer_context=self._optimizer_context,
            plugin_manager=self._plugin_manager,
            parent=self,
            plan_id=self._plan_ids,
        )

    def eval(self, value: Any) -> Any:  # noqa: ANN401
        """Evaluate the provided value as an expression.

        Evaluates an expression and returns the result based on the following
        rules:

        1. Non-string values are returned unchanged.
        2. Strings starting with `$$` will return the string with one `$`
           removed.
        3. Strings starting with `$` are evaluated as expressions, primarily
           intended to replace variable references with their values.
        4. Strings enclosed in `${{ ... }}` are evaluated as mathematical
           expressions, possibly embedding plan variable and plan function
           references prefixed by `$`.
        5. Strings enclosed in `$[[ ... ]]` are treated as templates: `$`
           prefixes or `${{ ... }}` expressions within the string are evaluated
           and interpolated.

        Args:
            value: The expression to evaluate, as a string or any other type.

        Returns:
            The evaluated result, which may vary in type depending on the evaluation context.
        """
        return (
            self._optimizer_context.expr.eval(value, self._vars)
            if isinstance(value, str)
            else value
        )

    def emit_event(self, event: Event) -> None:
        """Emit an event of the specified type with the provided data.

        When this method is called, the following steps are executed:

        1. All event handlers associated with the plan are invoked, which may
           modify the event.
        2. All observer functions registered for the specified event type via
           the `add_observer` method are called.
        3. If the plan was spawned from another plan, the `emit_event` method of
           the parent plan is also called.

        Args:
            event: The event object to emit.
        """
        if not event.plan_id:
            event.plan_id = self.plan_id
        for handler in self._handlers:
            event = handler.handle_event(event)
        if self._parent is None:
            self._optimizer_context.call_observers(event)
        else:
            self._parent.emit_event(event)

    def _check_condition(self, config: PlanStepConfig) -> bool:
        if config.if_ is not None:
            stripped = config.if_.strip()
            return (
                bool(self.eval(stripped))
                if stripped.startswith("${{") and stripped.endswith("}}")
                else bool(self.eval("${{" + stripped + "}}"))
            )
        return True

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        """Get the value of a plan variable.

        This method implements the `[]` operator on the plan object to retrieve
        the value associated with a specific plan variable.

        Args:
            name: The name of the variable whose value is to be retrieved.

        Returns:
            The value of the specified variable, which can be of any type.
        """
        if name in self._vars:
            return self._vars[name]
        msg = f"Unknown plan variable: `{name}`"
        raise AttributeError(msg)

    def _set_item(self, name: str, value: Any) -> None:  # noqa: ANN401
        if not name.isidentifier():
            msg = f"Not a valid variable name: `{name}`"
            raise AttributeError(msg)
        self._vars[name] = value

    def __setitem__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set a plan variable to the given value.

        This method implements the `[]` operator on the plan object to set the
        value of a specific plan variable.

        Args:
            name:  The name of the variable to set.
            value: The value to assign to the variable.
        """
        if name not in self._vars:
            msg = f"Unknown variable name: `{name}`"
            raise AttributeError(msg)
        self._set_item(name, value)

    def __contains__(self, name: str) -> bool:
        """Check if a variable exists.

        This method implements the `in` operator on the plan object to determine
        if a plan variable exists.

        Args:
            name: The name of the variable.

        Returns:
            `True` if the variable exists; otherwise, `False`.
        """
        return name in self._vars
