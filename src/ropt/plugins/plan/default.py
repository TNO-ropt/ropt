"""This module implements the default optimization plan plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from typing import Dict, Final, Type, Union

from ropt.config.plan import ContextConfig, StepConfig  # noqa: TCH001
from ropt.plan import Plan  # noqa: TCH001
from ropt.plugins.plan.base import ContextObj, PlanStep  # noqa: TCH001

from ._config import DefaultConfigContext
from ._evaluator import DefaultEvaluatorStep
from ._metadata import DefaultMetadataStep
from ._optimizer import DefaultOptimizerStep
from ._repeat import DefaultRepeatStep
from ._reset import DefaultResetStep
from ._setvar import DefaultSetStep
from ._tracker import DefaultTrackerContext
from ._update import DefaultUpdateStep
from .base import PlanPlugin

_CONTEXT_OBJECTS: Final[Dict[str, Type[ContextObj]]] = {
    "config": DefaultConfigContext,
    "tracker": DefaultTrackerContext,
}

_STEP_OBJECTS: Final[Dict[str, Type[PlanStep]]] = {
    "evaluator": DefaultEvaluatorStep,
    "metadata": DefaultMetadataStep,
    "optimizer": DefaultOptimizerStep,
    "repeat": DefaultRepeatStep,
    "reset": DefaultResetStep,
    "setvar": DefaultSetStep,
    "update": DefaultUpdateStep,
}


class DefaultPlanPlugin(PlanPlugin):
    """Default plan plugin class."""

    @singledispatchmethod
    def create(  # type: ignore[override]
        self,
        config: Union[ContextConfig, StepConfig],  # noqa: ARG002
        plan: Plan,  # noqa: ARG002
    ) -> Union[ContextObj, PlanStep]:
        """Initialize the plan plugin.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        msg = "Plan config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_context(self, config: ContextConfig, plan: Plan) -> ContextObj:
        _, _, name = config.init.lower().rpartition("/")
        obj = _CONTEXT_OBJECTS.get(name)
        if obj is not None:
            return obj(config, plan)

        msg = f"Unknown context object type: {config.init}"
        raise TypeError(msg)

    @create.register
    def _create_step(self, config: StepConfig, plan: Plan) -> PlanStep:
        _, _, step_name = config.run.lower().rpartition("/")
        step_obj = _STEP_OBJECTS.get(step_name)
        if step_obj is not None:
            return step_obj(config, plan)

        msg = f"Unknown step type: {config.run}"
        raise TypeError(msg)

    def is_supported(self, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return (method.lower() in _CONTEXT_OBJECTS) or (method.lower() in _STEP_OBJECTS)
