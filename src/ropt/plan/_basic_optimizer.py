"""This module defines a basic optimization object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from ropt.config.enopt import EnOptConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.plugins import PluginManager

from ._plan import Plan

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult
    from ropt.plan import Event
    from ropt.results import FunctionResults


@dataclass(slots=True)
class _Results:
    results: FunctionResults | None
    variables: NDArray[np.float64] | None
    exit_code: OptimizerExitCode = OptimizerExitCode.UNKNOWN


class BasicOptimizer:
    """A class for executing single optimization runs.

    The `BasicOptimizer` is designed to simplify the process of setting up and
    executing optimization workflows that consist primarily of a single
    optimization run. It offers a more streamlined approach compared to directly
    defining and managing a full `Plan` object, making it ideal for
    straightforward optimization tasks.

    This class provides a user-friendly interface for common optimization
    operations, including:

    - **Initiating a Single Optimization:**  Easily start an optimization
      process with a provided configuration and evaluator.
    - **Observing Optimization Events:** Register observer functions to monitor
      and react to various events that occur during the optimization, such as
      the start of an evaluation or the availability of new results.
    - **Abort Conditions:** Define a callback function that can be used to check
      for abort conditions during the optimization.
    - **Result Reporting:** Define a callback function that will be called
      whenever new results become available.
    - **Accessing Results:** After the optimization is complete, the optimal
      results, corresponding variables, and the optimization's exit code are
      readily accessible.
    - **Customizable Steps and Handlers:** While designed for single runs, it
      allows for the addition of custom plan steps and event handlers to the
      underlying `Plan` for more complex scenarios.

    By encapsulating the core elements of an optimization run, the
    `BasicOptimizer` reduces the boilerplate code required for simple
    optimization tasks, allowing users to focus on defining the optimization
    problem and analyzing the results.
    """

    def __init__(
        self,
        enopt_config: dict[str, Any] | EnOptConfig,
        evaluator: Callable[[NDArray[np.float64], EvaluatorContext], EvaluatorResult],
        *,
        constraint_tolerance: float = 1e-10,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize a `BasicOptimizer` object.

        This constructor sets up the necessary components for a single
        optimization run. It requires an optimization configuration and an
        evaluator, which together define the optimization problem and how to
        evaluate potential solutions. If a constraint value is within the
        `constraint_tolerance` of zero, it is considered satisfied. The `kwargs`
        may be used to define custom steps and event handlers to modify the
        behavior of the optimization process.

        Note: Custom  steps
            The optional keyword arguments (`kwargs`) provide a mechanism to
            inject a custom step into the optimization process. The behavior is
            as follows:

            1.  **Custom Step Execution:** If a single keyword argument is
                provided, the `BasicOptimizer` checks if a step with the same
                name exists. If so, that step is executed immediately, receiving
                the key-value pair as a keyword input, in addition to the
                evaluator function (via the `evaluator` keyword). Only one
                custom step can be executed this way, if other keyword arguments
                are present an error is raised. The custom step receives the
                `Plan` object and may return a custom function to execute.
            2.  **Default Optimization:** If no custom step is run, or if the
                custom step does not return a custom run function, the default
                optimization process is used.
            3.  **Callback Installation and Execution:** Finally, any callbacks
                added via `set_abort_callback` or `set_results_callback` are
                installed, and the appropriate run function is executed.

        Args:
            enopt_config:         The configuration for the optimization.
            evaluator:            The evaluator object.
            constraint_tolerance: The constraint violation tolerance.
            kwargs:               Optional keyword arguments.
        """
        self._config = EnOptConfig.model_validate(enopt_config)
        self._constraint_tolerance = constraint_tolerance
        self._evaluator = evaluator
        self._plugin_manager = PluginManager()
        self._observers: list[tuple[EventType, Callable[[Event], None]]] = []
        self._results: _Results
        self._kwargs: dict[str, Any] = kwargs

    @property
    def results(self) -> FunctionResults | None:
        """Return the optimal result found during the optimization.

        This property provides access to the best
        [`FunctionResults`][ropt.results.FunctionResults] object discovered
        during the optimization process. It encapsulates the objective function
        value, constraint values, and other relevant information about the
        optimal solution.

        Returns:
            The optimal result.
        """
        return self._results.results

    @property
    def variables(self) -> NDArray[np.float64] | None:
        """Return the optimal variables found during the optimization.

        This property provides access to the variable values that correspond to
        the optimal [`FunctionResults`][ropt.results.FunctionResults] object.
        These variables represent the solution that yielded the best objective
        function value found during the optimization process.

        Returns:
            The variables corresponding to the optimal result.
        """
        return self._results.variables

    @property
    def exit_code(self) -> OptimizerExitCode:
        """Return the exit code of the optimization run.

        This property provides access to the
        [`OptimizerExitCode`][ropt.enums.OptimizerExitCode] that indicates the
        outcome of the optimization process. It can be used to determine whether
        the optimization completed successfully, was aborted, or encountered an
        error.

        Returns:
            The exit code of the optimization run.
        """
        return self._results.exit_code

    def run(self) -> Self:
        """Run the optimization process.

        This method initiates the optimization workflow defined by the
        `BasicOptimizer` object. It executes the underlying `Plan`, which
        manages the optimization steps, result handling, and event processing.
        After the optimization is complete, the optimal results, variables, and
        exit code can be accessed via the corresponding properties.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """
        plan = Plan(self._plugin_manager)
        for event_type, function in self._observers:
            plan.add_event_handler(
                "observer", event_types={event_type}, callback=function
            )

        # Optionally run a custom step defined in the keyword arguments:
        custom_function: Callable[[Plan], OptimizerExitCode] | None = None
        key, value = next(iter(self._kwargs.items()), (None, None))
        if key is not None and self._plugin_manager.is_supported(
            "plan_step", method=key
        ):
            if len(self._kwargs) > 1:
                msg = "Only one custom step is allowed."
                raise TypeError(msg)
            custom_function = plan.add_step(key).run(
                evaluator=self._evaluator, **{key: value}
            )

        if custom_function is None:
            plan.add_evaluator("function_evaluator", evaluator=self._evaluator)
            optimizer = plan.add_step("optimizer")
            tracker = plan.add_event_handler(
                "tracker",
                constraint_tolerance=self._constraint_tolerance,
                sources={optimizer},
            )
            exit_code = optimizer.run(config=self._config)
            results = tracker["results"]
            variables = None if results is None else results.evaluations.variables
        else:
            exit_code = custom_function(plan)
            results = None
            variables = None

        self._results = _Results(
            results=results, variables=variables, exit_code=exit_code
        )

        return self

    def set_abort_callback(self, callback: Callable[[], bool]) -> Self:
        """Set a callback to check for abort conditions.

        The provided callback function will be invoked repeatedly during the
        optimization process. If the callback returns `True`, the optimization
        will be aborted, and the `BasicOptimizer` will exit with an
        [`OptimizerExitCode.USER_ABORT`][ropt.enums.OptimizerExitCode].

        The callback function should have no arguments and return a boolean
        value.

        Args:
            callback: The callable to check for abort conditions.

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
    ) -> Self:
        """Set a callback to report new results.

        The provided callback function will be invoked whenever new results
        become available during the optimization process. This allows for
        real-time monitoring and analysis of the optimization's progress.

        The required signature of the callback function should be:

        ```python
        def callback(results: tuple[FunctionResults, ...]) -> None:
            ...
        ```

        Args:
            callback: The callable that will be invoked to report new results.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """

        def _results_callback(event: Event) -> None:
            results = event.data.get("results", ())
            if event.config.transforms is not None:
                results = tuple(
                    item.transform_from_optimizer(event.config.transforms)
                    for item in results
                )
            callback(results)

        self._observers.append((EventType.FINISHED_EVALUATION, _results_callback))
        return self
