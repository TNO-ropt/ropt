"""This module implements the default optimization step plugin."""

from typing import Any, Set

from ropt.config.plan import StepConfig
from ropt.exceptions import ConfigError
from ropt.optimization import Plan, PlanContext

from .enopt_config import DefaultEnOptConfigStep
from .evaluator import DefaultEvaluatorStep
from .label import DefaultLabelStep
from .optimizer import DefaultOptimizerStep
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


def get_step(config: StepConfig, context: PlanContext, plan: Plan) -> Any:  # noqa: ANN401
    """Create a step object.

    Args:
        config:  The generic optimization step configuration
        context: The context of the optimization plan execution
        plan:    The plan that requires the step
    """
    assert config.model_extra is not None
    keys = set(config.model_extra.keys())
    if len(keys) > 1:
        msg = f"Step type is ambiguous: {keys}"
        raise ConfigError(msg)

    for key, factory in _FACTORIES.items():
        if key in keys:
            return factory(config.model_extra[key], context, plan)

    msg = f"Unknown step type: {keys.pop()}"
    raise TypeError(msg)


def get_default_steps() -> Set[str]:
    """Return the default step types.

    Returns:
        A set of supported step type names.
    """
    return set(_FACTORIES.keys())
