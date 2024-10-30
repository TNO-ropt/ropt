"""This module implements the default optimization plan plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from typing import Dict, Final, Type, Union

from ropt.config.plan import ResultHandlerConfig, RunStepConfig  # noqa: TCH001
from ropt.plan import Plan  # noqa: TCH001
from ropt.plugins.plan.base import (  # noqa: TCH001
    ResultHandler,
    RunStep,
)

from ._evaluator import DefaultEvaluatorStep
from ._metadata import DefaultMetadataHandler
from ._optimizer import DefaultOptimizerStep
from ._print import DefaultPrintStep
from ._repeat import DefaultRepeatStep
from ._set import DefaultSetStep
from ._table import DefaultTableHandler
from ._tracker import DefaultTrackerHandler
from .base import PlanPlugin

_STEP_OBJECTS: Final[Dict[str, Type[RunStep]]] = {
    "evaluator": DefaultEvaluatorStep,
    "optimizer": DefaultOptimizerStep,
    "print": DefaultPrintStep,
    "repeat": DefaultRepeatStep,
    "set": DefaultSetStep,
}

_RESULT_HANDLER_OBJECTS: Final[Dict[str, Type[ResultHandler]]] = {
    "metadata": DefaultMetadataHandler,
    "table": DefaultTableHandler,
    "tracker": DefaultTrackerHandler,
}


class DefaultPlanPlugin(PlanPlugin):
    """The default plan plugin class.

    This class provides a number of run steps and result handlers:

    `Run Steps`:
    : - A step that modifies one or more variables
        ([`optimizer`][ropt.plugins.plan._set.DefaultSetStep]).
    : - A step that performs a single ensemble evaluation
        ([`evaluator`][ropt.plugins.plan._evaluator.DefaultEvaluatorStep]).
    : - A step that performs a single ensemble evaluation
        ([`evaluator`][ropt.plugins.plan._evaluator.DefaultEvaluatorStep]).
    : - A step that prints a message to the console
        ([`print`][ropt.plugins.plan._print.DefaultPrintStep]).
    : - A step that repeats a number of steps
        ([`repeat`][ropt.plugins.plan._repeat.DefaultRepeatStep]).

    `Result Handlers`:
    : - A handler that adds metadata to results
        ([`metadata`][ropt.plugins.plan._metadata.DefaultMetadataHandler]).
    : - A handler that generates and saves tables of results
        ([`table`][ropt.plugins.plan._table.DefaultTableHandler]).
    : - A handler that tracks optimal results
        ([`tracker`][ropt.plugins.plan._tracker.DefaultTrackerHandler]).
    """

    @singledispatchmethod
    def create(  # type: ignore[override]
        self,
        config: Union[ResultHandlerConfig, RunStepConfig],
        plan: Plan,
    ) -> Union[ResultHandler, RunStep]:
        """Initialize the plan plugin.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        msg = "Plan config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_step(self, config: RunStepConfig, plan: Plan) -> RunStep:
        _, _, step_name = config.run.lower().rpartition("/")
        step_obj = _STEP_OBJECTS.get(step_name)
        if step_obj is not None:
            return step_obj(config, plan)

        msg = f"Unknown step type: {config.run}"
        raise TypeError(msg)

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

    def is_supported(self, method: str, *, explicit: bool) -> bool:  # noqa: ARG002
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return (method.lower() in _RESULT_HANDLER_OBJECTS) or (
            method.lower() in _STEP_OBJECTS
        )
