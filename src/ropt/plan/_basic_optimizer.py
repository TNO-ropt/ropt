"""This module defines a basic optimization object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Self

from ropt.config.enopt import EnOptConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted

from ._context import OptimizerContext
from ._plan import Plan

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import Evaluator
    from ropt.plan import Event
    from ropt.plugins.plan.base import ResultHandler
    from ropt.results import FunctionResults
    from ropt.transforms import OptModelTransforms


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
    """

    def __init__(
        self,
        enopt_config: dict[str, Any] | EnOptConfig,
        evaluator: Evaluator,
        *,
        transforms: OptModelTransforms | None = None,
        constraint_tolerance: float = 1e-10,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize an `BasicOptimizer` object.

        An optimization configuration and an evaluation object must be provided,
        as they define the optimization to perform.

        Args:
            enopt_config:         The configuration for the optimization.
            evaluator:            The evaluator object used to evaluate functions.
            transforms:           Optional transforms object.
            constraint_tolerance: The tolerance level used to detect constraint violations.
            kwargs:               Optional keywords that may be passed to optimization code.
        """
        self._config = EnOptConfig.model_validate(enopt_config)
        self._transforms = transforms
        self._constraint_tolerance = constraint_tolerance
        self._optimizer_context = OptimizerContext(evaluator=evaluator)
        self._observers: list[tuple[EventType, Callable[[Event], None]]] = []
        self._results: _Results
        self._metadata: dict[str, Any] = kwargs

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

    def run(self) -> Self:
        """Run the optimization.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """
        plan = Plan(self._optimizer_context)
        plan.add_function(self._run_func)
        for key, value in self._metadata.items():
            if plan.handler_exists(key):
                plan.add_handler(key, tags="__optimizer_tag__", **{key: value})
        for key, value in self._metadata.items():
            if plan.step_exists(key):
                step = plan.add_step(key)
                plan.run_step(step, **{key: value})
        for event_type, function in self._observers:
            self._optimizer_context.add_observer(event_type, function)
        result, exit_code = plan.run_function(self._transforms)
        self._results = _Results(
            results=None if result is None else result["results"],
            variables=None if result is None else result["variables"],
            exit_code=exit_code,
        )
        return self

    def _run_func(
        self, plan: Plan, transforms: OptModelTransforms | None
    ) -> tuple[ResultHandler | None, OptimizerExitCode | None]:
        optimizer = plan.add_step(
            "optimizer",
            config=self._config,
            transforms=transforms,
            tags="__optimizer_tag__",
        )
        tracker = plan.add_handler(
            "tracker",
            constraint_tolerance=self._constraint_tolerance,
            tags="__optimizer_tag__",
        )
        plan.run_step(optimizer)
        return tracker, optimizer["exit_code"]

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
