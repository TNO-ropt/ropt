"""This module implements the default optimization plan plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from typing import Dict, Final, Type, Union

from ropt.config.plan import PlanStepConfig, ResultHandlerConfig
from ropt.plan import Plan
from ropt.plugins.plan.base import PlanStep, ResultHandler

from ._evaluator import DefaultEvaluatorStep
from ._load_data import DefaultLoadStep
from ._metadata import DefaultMetadataHandler
from ._optimizer import DefaultOptimizerStep
from ._print import DefaultPrintStep
from ._repeat import DefaultRepeatStep
from ._save_data import DefaultSaveStep
from ._save_results import DefaultSaveHandler
from ._set import DefaultSetStep
from ._table import DefaultTableHandler
from ._tracker import DefaultTrackerHandler
from .base import PlanPlugin

_STEP_OBJECTS: Final[Dict[str, Type[PlanStep]]] = {
    "evaluator": DefaultEvaluatorStep,
    "load": DefaultLoadStep,
    "optimizer": DefaultOptimizerStep,
    "save": DefaultSaveStep,
    "print": DefaultPrintStep,
    "repeat": DefaultRepeatStep,
    "set": DefaultSetStep,
}

_RESULT_HANDLER_OBJECTS: Final[Dict[str, Type[ResultHandler]]] = {
    "metadata": DefaultMetadataHandler,
    "save": DefaultSaveHandler,
    "table": DefaultTableHandler,
    "tracker": DefaultTrackerHandler,
}


class DefaultPlanPlugin(PlanPlugin):
    """The default plan plugin class.

    This class provides a number of steps and result handlers:

    `Steps`:
    : - A step that modifies one or more variables
        ([`set`][ropt.plugins.plan._set.DefaultSetStep]).
    : - A step that loads data from a file
        ([`load`][ropt.plugins.plan._load_data.DefaultLoadStep]).
    : - A step that saves data to a file
        ([`save`][ropt.plugins.plan._save_data.DefaultSaveStep]).
    : - A step that performs a single ensemble evaluation
        ([`evaluator`][ropt.plugins.plan._evaluator.DefaultEvaluatorStep]).
    : - A step that runs an optimization
        ([`optimizer`][ropt.plugins.plan._optimizer.DefaultOptimizerStep]).
    : - A step that prints a message to the console
        ([`print`][ropt.plugins.plan._print.DefaultPrintStep]).
    : - A step that repeats a number of steps
        ([`repeat`][ropt.plugins.plan._repeat.DefaultRepeatStep]).

    `Result Handlers`:
    : - A handler that adds metadata to results
        ([`metadata`][ropt.plugins.plan._metadata.DefaultMetadataHandler]).
    : - A handler that tracks optimal results
        ([`tracker`][ropt.plugins.plan._tracker.DefaultTrackerHandler]).
    : - A handler that generates and saves tables of results
        ([`table`][ropt.plugins.plan._table.DefaultTableHandler]).
    : - A handler that saves results to netCDF files
        ([`save`][ropt.plugins.plan._save_results.DefaultSaveHandler]).
    """

    @singledispatchmethod
    def create(  # type: ignore[override]
        self,
        config: Union[PlanStepConfig, ResultHandlerConfig],
        plan: Plan,
    ) -> Union[ResultHandler, PlanStep]:
        """Initialize the plan plugin.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        msg = "Plan config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_step(self, config: PlanStepConfig, plan: Plan) -> PlanStep:
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
