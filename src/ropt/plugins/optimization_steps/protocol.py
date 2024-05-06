"""This module implements the protocol to be followed by step plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Protocol

from ropt.plugins.protocol import PluginProtocol

if TYPE_CHECKING:
    from ropt.optimization import Plan, PlanContext


class OptimizationStepsProtocol(Protocol):
    """Protocol optimization step plugins."""

    def __init__(self, context: PlanContext, plan: Plan) -> None:
        """Create a default optimization step plugin.

        Args:
            context: The context of the running plan.
            plan:    The current plan.
        """

    def get_step(self, config: Dict[str, Any]) -> Any:  # noqa: ANN401
        """Get a step object.

        Args:
            config:  The generic optimization step configuration
            context: The context of the optimization plan execution
            plan:    The plan that requires the step
        """


class OptimizationStepsPluginProtocol(PluginProtocol, Protocol):
    """Optimization step plugin protocol."""

    def create(self, context: PlanContext, plan: Plan) -> OptimizationStepsProtocol:
        """Create an optimization step.

        Args:
            context: The context of the running plan.
            plan:    The current plan.
        """
