"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.plugins import PluginManager

if TYPE_CHECKING:
    from ropt.evaluator import Evaluator


class OptimizerContext:
    """Manages shared state and resources for an optimization plan.

    The OptimizerContext acts as a central hub for managing shared resources and
    state across all steps within an optimization plan. This ensures that
    different parts of the plan can access and interact with the same
    information and tools.

    This context object is responsible for:

    - Providing a callable `Evaluator` for evaluating functions, which is
      essential for optimization algorithms to assess the quality of solutions.
      The evaluator is used to calculate the objective function's value and
      potentially constraint values for given variables.
    - Managing a `PluginManager` to retrieve and utilize plugins, allowing for
      extensibility and customization of the optimization workflow. Plugins are
      modular pieces of code that extend the functionality of the optimization
      framework, such as new `PlanStep` or `PlanHandler` implementations.

    Args:
        evaluator:      A callable for evaluating functions in the plan.
        plugin_manager: A plugin manager; a default is created if not provided.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        """Initializes the optimization context.

        Sets up the shared state and resources required for an optimization
        plan. This includes a function evaluator and a plugin manager.

        Args:
            evaluator:      A callable for evaluating functions in the plan.
            plugin_manager: A plugin manager; a default is created if not provided.
        """
        self.evaluator = evaluator
        self.plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )
