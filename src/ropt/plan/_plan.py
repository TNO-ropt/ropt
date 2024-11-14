"""This module defines the optimization plan object."""

from __future__ import annotations

from copy import deepcopy
from itertools import chain, count
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from ropt.config.plan import PlanConfig, PlanStepConfig
    from ropt.plan import Event
    from ropt.plugins.plan.base import PlanStep, ResultHandler

    from ._context import OptimizerContext


class Plan:
    """The plan class for executing optimization workflows."""

    def __init__(
        self,
        config: PlanConfig,
        optimizer_context: OptimizerContext,
        parent: Optional[Plan] = None,
        plan_id: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize a plan object.

        This method initializes a plan using a `PlanConfig` object and an
        [`OptimizerContext`][ropt.plan.OptimizerContext] object. An optional
        [`plugin_manager`][ropt.plugins.PluginManager] argument allows for the
        specification of custom plugins for result handlers and step objects
        within the plan. If omitted, only plugins installed through Python's
        standard entry points are used.

        Plans can spawn additional plans within their workflow, with each
        spawned plan receiving a reference to its parent plan via the `parent`
        argument.

        Plan hierarchies are tracked using the `plan_id` attribute, a tuple of
        integers. A directly created plan has `plan_id == (0,)`. When a plan
        spawns a new plan with the [`spawn`][ropt.plan.Plan.spawn] method, the
        child receives an incremental index appended to its parent's `plan_id`,
        forming a unique sequence that reflects both the order of creation and
        nesting. This structure enables efficient tracing across both sequential
        and nested plan workflows.

        If the `optimizer_context` objects has variables defined, these are
        copied into the plan and available under the name they were added to the
        context. These variables can then be modified, but when spawning new
        plans, the new plan will have the original values for these variables.

        Args:
            config:            The configuration for the optimizer.
            optimizer_context: The context in which the plan will execute,
                               providing shared resources across steps.
            parent:            Optional reference to the parent plan that
                               spawned this plan.
            plan_id:           The ID of the plan, reflecting its hierarchy.
        """
        self._plan_config = config
        self._optimizer_context = optimizer_context
        self._vars: Dict[str, Any] = deepcopy(optimizer_context.variables)
        self._plan_id: Tuple[int, ...] = (0,) if plan_id is None else plan_id
        self._spawn_id: int = -1
        self._result_id_iter = count()

        self._set_item("plan_id", list(self.plan_id))
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
        self._steps = self.create_steps(config.steps)
        self._handlers: List[ResultHandler] = [
            self._optimizer_context.plugin_manager.get_plugin(
                "plan", method=config.run
            ).create(config, self)
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
    def aborted(self) -> bool:
        """Check if the plan was aborted by the user.

        Returns:
            bool: `True` if the plan was aborted by the user; otherwise, `False`.
        """
        return self._aborted

    @property
    def plan_id(self) -> Tuple[int, ...]:
        """Return the ID of the plan.

        Each plan has a unique ID, stored as a tuple of integers, which reflects
        both its creation order and any nesting structure within the plan
        hierarchy. When a plan spawns additional plans, this hierarchy is
        encoded in the `plan_id` attribute as a sequential tuple.

        Returns:
            tuple: The unique tuple-based ID for this plan.
        """
        return self._plan_id

    @property
    def result_id_iterator(self) -> Iterator[int]:
        """Return the iterator for result IDs.

        This iterator generates consecutive unique IDs for results produced by
        steps within the plan, ensuring each result can be distinctly
        identified.

        Returns:
            Iterator: An iterator that yields unique result IDs.
        """
        return self._result_id_iter

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
            self._optimizer_context.plugin_manager.get_plugin(
                "plan", method=step_config.run
            ).create(step_config, self)
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
        self._spawn_id += 1
        return Plan(
            config,
            optimizer_context=self._optimizer_context,
            parent=self,
            plan_id=(*self._plan_id, self._spawn_id),
        )

    def eval(self, value: Any) -> Any:  # noqa: ANN401
        """Evaluate the provided value as an expression.

        If the value is not a string, it is returned unchanged, otherwise it is
        evaluated as an expression. Refer to the
        [`eval`][ropt.plan.ExpressionEvaluator.eval] method of the
        [`ExpressionEvaluator`][ropt.plan.ExpressionEvaluator] class for more
        details.

        Args:
            value: The expression to evaluate, as a string or any other type.

        Returns:
            The evaluated result, which may vary in type depending on the context.
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
                if stripped.startswith("$(") and stripped.endswith(")")
                else bool(self.eval("$(" + stripped + ")"))
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
