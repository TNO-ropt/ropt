"""This module implements the default optimization plan plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from typing import Dict, Final, Type, Union

from ropt.config.plan import ResultHandlerConfig, StepConfig  # noqa: TCH001
from ropt.plan import Plan  # noqa: TCH001
from ropt.plugins.plan.base import PlanStep, ResultHandler  # noqa: TCH001

from ._copy import DefaultCopyStep
from ._evaluator import DefaultEvaluatorStep
from ._metadata import DefaultMetadataHandler
from ._optimizer import DefaultOptimizerStep
from ._repeat import DefaultRepeatStep
from ._set import DefaultSetStep
from ._table import DefaultTableHandler
from ._tracker import DefaultTrackerHandler
from .base import PlanPlugin

_RESULT_HANDLER_OBJECTS: Final[Dict[str, Type[ResultHandler]]] = {
    "metadata": DefaultMetadataHandler,
    "table": DefaultTableHandler,
    "tracker": DefaultTrackerHandler,
}

_STEP_OBJECTS: Final[Dict[str, Type[PlanStep]]] = {
    "copy": DefaultCopyStep,
    "evaluator": DefaultEvaluatorStep,
    "optimizer": DefaultOptimizerStep,
    "repeat": DefaultRepeatStep,
    "set": DefaultSetStep,
}


class DefaultPlanPlugin(PlanPlugin):
    """Default plan plugin class."""

    @singledispatchmethod
    def create(  # type: ignore[override]
        self,
        config: Union[ResultHandlerConfig, StepConfig],  # noqa: ARG002
        plan: Plan,  # noqa: ARG002
    ) -> Union[ResultHandler, PlanStep]:
        """Initialize the plan plugin.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        msg = "Plan config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_result_handler(
        self, config: ResultHandlerConfig, plan: Plan
    ) -> ResultHandler:
        _, _, name = config.run.lower().rpartition("/")
        obj = _RESULT_HANDLER_OBJECTS.get(name)
        if obj is not None:
            return obj(config, plan)

        msg = f"Unknown results handler object type: {config.run}"
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
        return (method.lower() in _RESULT_HANDLER_OBJECTS) or (
            method.lower() in _STEP_OBJECTS
        )
