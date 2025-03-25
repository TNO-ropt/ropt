"""This module defines a basic optimization object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Self

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
        **kwargs: Any,  # noqa: ANN401
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
        self._kwargs: dict[str, Any] = kwargs

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

    def run(self) -> Self:
        """Run the optimization.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """

        def _run_func(
            plan: Plan, transforms: OptModelTransforms | None
        ) -> tuple[ResultHandler | None, OptimizerExitCode | None]:
            exit_code = plan.run_step(
                optimizer, config=self._config, transforms=transforms
            )
            return plan.get(tracker, "results"), exit_code

        plan = Plan(self._optimizer_context)

        for key, value in self._kwargs.items():
            if plan.step_exists(key):
                plan.run_step(plan.add_step(key), **{key: value})

        if not plan.has_function():
            optimizer = plan.add_step("optimizer")
            tracker = plan.add_handler(
                "tracker",
                constraint_tolerance=self._constraint_tolerance,
                sources={optimizer},
            )
            plan.add_function(_run_func)
            for key, value in self._kwargs.items():
                if plan.handler_exists(key):
                    plan.add_handler(key, sources={optimizer}, **{key: value})

        for event_type, function in self._observers:
            self._optimizer_context.add_observer(event_type, function)

        results, exit_code = plan.run_function(self._transforms)
        variables = None if results is None else results.evaluations.variables
        self._results = _Results(
            results=results,
            variables=variables,
            exit_code=exit_code,
        )
        return self

    def set_abort_callback(self, callback: Callable[[], bool]) -> Self:
        """Set an abort callback.

        The callback will be called repeated during optimization. If it returns `True`,
        the optimization will be aborted.

        Args:
            callback: The callable that will be used to check for abort conditions.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """

        def _check_abort_callback(_: Event) -> None:
            if callback():
                raise OptimizationAborted(exit_code=OptimizerExitCode.USER_ABORT)

        self._observers.append((EventType.START_EVALUATION, _check_abort_callback))
        return self

    def set_results_callback(
        self,
        callback: Callable[..., None],
        *,
        transformed: bool = False,
    ) -> Self:
        """Set a callback to report results.

        The callback will be called whenever new results become available.

        If `transformed` is `False` the callback should have this signature:

            `def callback(results: tuple[Results, ...]) -> None: ...`

        If `transformed` is `True`, the signature should be:

            def callback(
                results: tuple[Results, ...], transformed_results: tuple[Results, ...]
            ) -> None: ...`

        Args:
            callback:    The callable that will be used to report results.
            transformed: If true also pass the transformed results.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """

        def _results_callback(event: Event) -> None:
            results = event.data.get("results", ())
            if transformed:
                transformed_results = event.data.get("transformed_results", ())
                if results or transformed_results:
                    callback(results, transformed_results)
            elif results:
                callback(results)

        self._observers.append((EventType.FINISHED_EVALUATION, _results_callback))
        return self
