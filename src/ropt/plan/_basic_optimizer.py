"""This module defines a basic optimization object."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Self

import numpy as np

from ropt.config.enopt import EnOptConfig
from ropt.config.plan import PlanConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted

from ._context import OptimizerContext
from ._plan import Plan

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.evaluator import Evaluator
    from ropt.plan import Event
    from ropt.results import FunctionResults


@dataclass(slots=True)
class _Results:
    results: FunctionResults | None
    variables: NDArray[np.float64] | None
    exit_code: OptimizerExitCode = OptimizerExitCode.UNKNOWN


class BasicOptimizer:
    """A class for running optimization plans.

    `BasicOptimizer` objects are designed for use cases where the optimization
    workflow comprises a single optimization run. Using this object can be more
    convenient than defining and running an optimization plan directly in such
    cases.

    This class provides the following features:

    - Start a single optimization.
    - Add observer functions connected to various optimization events.
    - Attach metadata to each result generated during the optimization.
    - Generate tables summarizing the optimization results.
    """

    def __init__(
        self,
        enopt_config: dict[str, Any] | EnOptConfig,
        evaluator: Evaluator,
        *,
        constraint_tolerance: float = 1e-10,
    ) -> None:
        """Initialize an `BasicOptimizer` object.

        An optimization configuration and an evaluation object must be provided,
        as they define the optimization to perform.

        Args:
            enopt_config:         The configuration for the optimization.
            evaluator:            The evaluator object used to evaluate functions.
            constraint_tolerance: The tolerance level used to detect constraint violations.
        """
        self._config = EnOptConfig.model_validate(enopt_config)
        self._constraint_tolerance = constraint_tolerance
        self._optimizer_context = OptimizerContext(evaluator=evaluator)
        self._observers: list[tuple[EventType, Callable[[Event], None]]] = []
        self._results: _Results
        self._event_data: dict[str, Any] = {}

    @property
    def results(self) -> FunctionResults | None:
        """Return the optimal result.

        Returns:
            The optimal result found during optimization.
        """
        return self._results.results

    @property
    def variables(self) -> NDArray[np.float64] | None:
        """Return the optimal variables.

        Returns:
            The variables corresponding to the optimal result.
        """
        return self._results.variables

    @property
    def exit_code(self) -> OptimizerExitCode:
        """Return the exit code.

        Returns:
            The exit code of the optimization run.
        """
        return self._results.exit_code

    def add_observer(
        self, event_type: EventType, function: Callable[[Event], None]
    ) -> Self:
        """Add an observer.

        Observers are callables that are triggered when an optimization event
        occurs. This method adds an observer that responds to a specified event
        type.

        Args:
            event_type: The type of event to observe.
            function:   The callable to invoke when the event is emitted.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """
        self._observers.append((event_type, function))
        return self

    def add_event_data(self, data: dict[str, Any]) -> Self:
        """Add data that will be merged into event data.

        The given data will be merged into the event data dictionary that
        is passed via events emitted by the optimizer.

        Args:
            data: The data to add.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """
        self._event_data.update(deepcopy(data))
        return self

    def run(self) -> Self:
        """Run the optimization.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """
        steps: list[dict[str, Any]] = []
        if np.any(self._config.objectives.auto_scale) or (
            self._config.nonlinear_constraints is not None
            and np.any(self._config.nonlinear_constraints.auto_scale)
        ):
            steps.append(
                {
                    "run": "evaluator",
                    "with": {
                        "config": "$__config__",
                        "tags": "__optimizer_tag__",
                        "data": self._event_data,
                        "values": self._config.variables.initial_values,
                    },
                }
            )
        steps.append(
            {
                "run": "optimizer",
                "with": {
                    "config": "$__config__",
                    "exit_code_var": "__exit_code__",
                    "tags": "__optimizer_tag__",
                    "data": self._event_data,
                },
            }
        )
        plan = Plan(
            PlanConfig.model_validate(
                {
                    "variables": {
                        "__config__": self._config,
                        "__optimum_tracker__": None,
                        "__exit_code__": OptimizerExitCode.UNKNOWN,
                    },
                    "steps": steps,
                    "handlers": [
                        {
                            "run": "tracker",
                            "with": {
                                "var": "__optimum_tracker__",
                                "constraint_tolerance": self._constraint_tolerance,
                                "tags": "__optimizer_tag__",
                            },
                        },
                    ],
                }
            ),
            self._optimizer_context,
        )
        for event_type, function in self._observers:
            self._optimizer_context.add_observer(event_type, function)
        plan.run()
        results = plan["__optimum_tracker__"]
        self._results = _Results(
            results=results,
            variables=None if results is None else results.evaluations.variables,
            exit_code=plan["__exit_code__"],
        )
        return self

    @staticmethod
    def abort_optimization() -> NoReturn:
        """Abort the current optimization run.

        This method can be called from within callbacks to interrupt the ongoing
        optimization plan. The exact point at which the optimization is aborted
        depends on the step that is executing at that point. For example, within
        a running optimizer, the process will be interrupted after completing
        the current function evaluation.
        """
        raise OptimizationAborted(exit_code=OptimizerExitCode.USER_ABORT)
