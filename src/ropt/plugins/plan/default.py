"""This module implements the default optimization plan plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from typing import Dict, Final, Type, Union

from ropt.config.plan import EventHandlerConfig, StepConfig  # noqa: TCH001
from ropt.plan import Plan  # noqa: TCH001
from ropt.plugins.plan.base import EventHandler, PlanStep  # noqa: TCH001

from ._evaluator import DefaultEvaluatorStep
from ._metadata import DefaultMetadataHandler
from ._optimizer import DefaultOptimizerStep
from ._repeat import DefaultRepeatStep
from ._setvar import DefaultSetStep
from ._tracker import DefaultTrackerHandler
from .base import PlanPlugin

_HANDLER_OBJECTS: Final[Dict[str, Type[EventHandler]]] = {
    "tracker": DefaultTrackerHandler,
    "metadata": DefaultMetadataHandler,
}

_STEP_OBJECTS: Final[Dict[str, Type[PlanStep]]] = {
    "evaluator": DefaultEvaluatorStep,
    "optimizer": DefaultOptimizerStep,
    "repeat": DefaultRepeatStep,
    "setvar": DefaultSetStep,
}


class DefaultPlanPlugin(PlanPlugin):
    """Default plan plugin class."""

    @singledispatchmethod
    def create(  # type: ignore[override]
        self,
        config: Union[EventHandlerConfig, StepConfig],  # noqa: ARG002
        plan: Plan,  # noqa: ARG002
    ) -> Union[EventHandler, PlanStep]:
        """Initialize the plan plugin.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        msg = "Plan config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_handler(self, config: EventHandlerConfig, plan: Plan) -> EventHandler:
        _, _, name = config.init.lower().rpartition("/")
        obj = _HANDLER_OBJECTS.get(name)
        if obj is not None:
            return obj(config, plan)

        msg = f"Unknown event handler object type: {config.init}"
        raise TypeError(msg)

    @create.register
    def _create_step(self, config: StepConfig, plan: Plan) -> PlanStep:
        _, _, step_name = config.run.lower().rpartition("/")
        step_obj = _STEP_OBJECTS.get(step_name)
        if step_obj is not None:
            return step_obj(config, plan)

        msg = f"Unknown step type: {config.run}"
        raise TypeError(msg)

    def is_supported(self, method: str, *, explicit: bool) -> bool:  # noqa: ARG002
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return (method.lower() in _HANDLER_OBJECTS) or (method.lower() in _STEP_OBJECTS)
