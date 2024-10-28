"""This module defines the optimization plan object."""

from __future__ import annotations

from itertools import chain, count
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from numpy.random import default_rng

from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.config.plan import RunStepConfig, SetStepConfig
from ropt.enums import EventType
from ropt.plugins import PluginManager

from ._expr import ExpressionEvaluator
from ._set import SetStep

if TYPE_CHECKING:
    from ropt.config.plan import PlanConfig
    from ropt.evaluator import Evaluator
    from ropt.plan import Event, ResultHandler, RunStep


class OptimizerContext:
    """Context class for shared state across a plan.

    An optimizer context object holds the information and state shared across
    all steps in an optimization plan. This currently includes the following:

    - An [`Evaluator`][ropt.evaluator.Evaluator] callable for evaluating
      functions.
    - A seed for a random number generator used in stochastic gradient
      estimation.
    - An iterator that generates unique result IDs.
    """

    def __init__(self, evaluator: Evaluator, seed: Optional[int] = None) -> None:
        """Initialize the optimization context.

        Args:
            evaluator:      The callable for running function evaluations
            seed:           Optional seed for the random number generator
        """
        self.evaluator = evaluator
        self.rng = default_rng(DEFAULT_SEED) if seed is None else default_rng(seed)
        self.result_id_iter = count()


class Plan:
    """The plan class for executing optimization workflows."""

    def __init__(
        self,
        config: PlanConfig,
        optimizer_context: OptimizerContext,
        plugin_manager: Optional[PluginManager] = None,
        parent: Optional[Plan] = None,
    ) -> None:
        """Initialize a plan object.

        The plan requires a `PlanConfig` object and an `OptimizationContext`
        object. An optional `plugin_manager` argument allows specification of
        custom plugins for result handlers and step objects within the plan. If
        omitted, the plan will use only plugins installed through Python's
        standard entry points mechanism.

        Plans may create other plans to be used within their workflow. If this
        happens, the new plan will receive a reference to the original plan via
        the `parent` argument.

        Args:
            config:            The configuration for the optimizer.
            optimizer_context: The context in which the plan will execute,
                               providing shared resources across steps.
            plugin_manager:    Optional custom plugin manager for step and
                               result handler plugins.
            parent:            Optional parent plan that spawned this plan.
        """
        self._plan_config = config
        self._optimizer_context = optimizer_context
        self._vars: Dict[str, Any] = {}
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            event: [] for event in EventType
        }

        self._plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )
        for var in chain(
            self._plan_config.inputs,
            self._plan_config.outputs,
            self._plan_config.variables,
        ):
            if var in self._vars:
                msg = f"Variable already exists: `{var}"
                raise AttributeError(msg)
            self._set_item(var, None)
        self._steps = self.create_steps(config.steps)
        self._handlers: List[ResultHandler] = [
            self._plugin_manager.get_plugin("plan", method=config.run).create(
                config, self
            )
            for config in config.results
        ]
        self._aborted = False
        self._parent = parent
        self._expression_evaluator = ExpressionEvaluator(self._vars)

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
    def aborted(self) -> bool:
        """Check if the plan was aborted by the user.

        Returns:
            bool: `True` if the plan was aborted by the user; otherwise, `False`.
        """
        return self._aborted

    def create_steps(
        self, step_configs: List[Union[RunStepConfig, SetStepConfig]]
    ) -> List[Union[RunStep, SetStep]]:
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
            if isinstance(step_config, RunStepConfig)
            else SetStep(step_config, self)
            for step_config in step_configs
        ]

    def run_steps(self, steps: List[Union[SetStep, RunStep]]) -> None:
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
            if isinstance(task, SetStep) or self._check_condition(task.step_config):
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
        return Plan(
            config,
            optimizer_context=self._optimizer_context,
            plugin_manager=self._plugin_manager,
            parent=self,
        )

    def eval(self, value: Any) -> Any:  # noqa: ANN401
        """Evaluate the provided value as an expression.

        Evaluates an expression and returns the result based on the following
        rules:

        1. Non-string values are returned unchanged.
        2. Strings starting with `$` are evaluated as expressions, primarily
           replacing variable references with their values.
        3. Strings starting with `$$` will return the string with one `$`
           removed.
        4. Strings enclosed in `{{ ... }}` are evaluated as mathematical
           expressions, allowing variable references prefixed by `$`.
        5. Strings enclosed in `[[ ... ]]` are treated as templates: `$`
           prefixes or `{{ ... }}` expressions within the string are evaluated
           and interpolated.

        Note:
            The result of mathematical expressions is restricted to numerical
            types, lists, and NumPy arrays. However, plan variables may contain
            values of any type, so expressions of the form `$identifier` may
            evaluate to values of varying types.

        Args:
            value: The expression to evaluate, as a string or any other type.

        Returns:
            The evaluated result, which may vary in type depending on the evaluation context.
        """
        return (
            self._expression_evaluator.eval(value) if isinstance(value, str) else value
        )

    def add_observer(
        self,
        event: EventType,
        callback: Callable[[Event], None],
    ) -> None:
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
        """
        self._subscribers[event].append(callback)

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
        for handler in self._handlers:
            event = handler.handle_event(event)
        for callback in self._subscribers[event.event_type]:
            callback(event)
        if self._parent is not None:
            self._parent.emit_event(event)

    def _check_condition(self, config: RunStepConfig) -> bool:
        if config.if_ is not None:
            stripped = config.if_.strip()
            return (
                bool(self.eval(stripped))
                if stripped.startswith("{{") and stripped.endswith("}}")
                else bool(self.eval("{{" + stripped + "}}"))
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
