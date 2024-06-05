"""This module implements the abstract base class for step plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.optimization import Plan, PlanContext


class OptimizationSteps(ABC):
    """Abstract base for optimization step plugins."""

    @abstractmethod
    def __init__(self, context: PlanContext, plan: Plan) -> None:
        """Create a default optimization step plugin.

        Args:
            context: The context of the running plan.
            plan:    The current plan.
        """

    @abstractmethod
    def get_step(self, config: Dict[str, Any]) -> Any:  # noqa: ANN401
        """Get a step object.

        Args:
            config:  The generic optimization step configuration
            context: The context of the optimization plan execution
            plan:    The plan that requires the step
        """


class OptimizationStepsPlugin(Plugin):
    """Optimization step plugin base."""

    @abstractmethod
    def create(self, context: PlanContext, plan: Plan) -> OptimizationSteps:
        """Create an optimization step.

        Args:
            context: The context of the running plan.
            plan:    The current plan.
        """
