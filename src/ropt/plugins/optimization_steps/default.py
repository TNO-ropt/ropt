"""This module implements the default optimization step plugin."""

from typing import Any, Dict

from ropt.exceptions import ConfigError
from ropt.optimization import Plan, PlanContext

from .enopt_config import DefaultEnOptConfigStep
from .evaluator import DefaultEvaluatorStep
from .label import DefaultLabelStep
from .optimizer import DefaultOptimizerStep
from .protocol import OptimizationStepsPluginProtocol, OptimizationStepsProtocol
from .reset_tracker import DefaultResetTrackerStep
from .restart import DefaultRestartStep
from .tracker import DefaultTrackerStep
from .update_config import DefaultUpdateConfigStep

_FACTORIES = {
    "reset_tracker": DefaultResetTrackerStep,
    "label": DefaultLabelStep,
    "config": DefaultEnOptConfigStep,
    "update_config": DefaultUpdateConfigStep,
    "tracker": DefaultTrackerStep,
    "restart": DefaultRestartStep,
    "optimizer": DefaultOptimizerStep,
    "evaluator": DefaultEvaluatorStep,
}


class DefaultOptimizationSteps(OptimizationStepsProtocol):
    """Default plugin for optimization steps."""

    def __init__(self, context: PlanContext, plan: Plan) -> None:
        """Create a default optimization step plugin.

        Args:
            context: The context of the running plan.
            plan:    The current plan.
        """
        self._context = context
        self._plan = plan

    def get_step(self, config: Dict[str, Any]) -> Any:  # noqa: ANN401
        """Get a step object.

        Args:
            config:  The generic optimization step configuration
            context: The context of the optimization plan execution
            plan:    The plan that requires the step
        """
        keys = set(config.keys())
        if len(keys) > 1:
            msg = f"Step type is ambiguous: {keys}"
            raise ConfigError(msg)
        key = keys.pop()

        _, _, step_name = key.lower().rpartition("/")
        factory = _FACTORIES.get(step_name)
        if factory is not None:
            return factory(config[key], self._context, self._plan)

        msg = f"Unknown step type: {key}"
        raise TypeError(msg)


class DefaultOptimizationStepsPlugin(OptimizationStepsPluginProtocol):
    """Default filter transform plugin class."""

    def create(self, context: PlanContext, plan: Plan) -> DefaultOptimizationSteps:
        """Initialize the realization filter plugin.

        See the [ropt.plugins.optimization_steps.protocol.OptimizationStepsPlugin][] protocol.

        # noqa
        """
        return DefaultOptimizationSteps(context, plan)

    def is_supported(self, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.protocol.Plugin][] protocol.

        # noqa
        """
        return method.lower() in _FACTORIES
