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
    from ropt.plugins.plan.base import PlanHandler
    from ropt.results import FunctionResults
    from ropt.transforms import OptModelTransforms


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
      allows for the addition of custom steps and handlers to the underlying
      `Plan` for more complex scenarios. It is possible to pass keyword
      arguments to the custom steps and handlers.

    By encapsulating the core elements of an optimization run, the
    `BasicOptimizer` reduces the boilerplate code required for simple
    optimization tasks, allowing users to focus on defining the optimization
    problem and analyzing the results.
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
        """Initialize a `BasicOptimizer` object.

        This constructor sets up the necessary components for a single
        optimization run. It requires an optimization configuration and an
        evaluator, which together define the optimization problem and how to
        evaluate potential solutions. The `transforms` object can be used to
        apply transformations to the optimization model, such as scaling or
        shifting variables. If a constraint value is within the
        `constraint_tolerance` of zero, it is considered satisfied. The `kwargs`
        may be used to define custom steps, and handlers to modify the behavior
        of the optimization process.

        Note: Custom  steps
            The optional keyword arguments (`kwargs`) provide a mechanism to
            inject a custom step into the optimization process. The behavior is
            as follows:

            1.  **Custom Step Execution:** If a single keyword argument is
                provided, the `BasicOptimizer` checks if a step with the same
                name exists. If so, that step is executed immediately, receiving
                the key-value pair as a keyword input, together with the
                transforms passed via a `transforms` keyword. Only one custom
                step can be executed this way, if other keyword arguments are
                present an error is raised. The custom step receives the `Plan`
                object and may install a custom run function to be executed
                later, or install custom result handlers.
            2.  **Default Optimization:** If no custom step is run, or if the
                custom step does not install a custom run function, the default
                optimization process is used.
            3.  **Callback Installation and Execution:** Finally, any callbacks
                added via `set_abort_callback` or `set_results_callback` are
                installed, and the appropriate run function is executed.

        Args:
            enopt_config:         The configuration for the optimization.
            evaluator:            The evaluator object.
            transforms:           Optional transforms.
            constraint_tolerance: The constraint violation tolerance.
            kwargs:               Optional keyword arguments.
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

        def _run_func(
            plan: Plan, transforms: OptModelTransforms | None
        ) -> tuple[PlanHandler | None, OptimizerExitCode | None]:
            exit_code = plan.run_step(
                optimizer, config=self._config, transforms=transforms
            )
            return plan.get(tracker, "results"), exit_code

        plan = Plan(self._optimizer_context)

        # Optionally run a custom step defined in the keyword arguments:
        key, value = next(iter(self._kwargs.items()), (None, None))
        if key is not None and plan.step_exists(key):
            if len(self._kwargs) > 1:
                msg = "Only one custom step is allowed."
                raise TypeError(msg)
            plan.run_step(
                plan.add_step(key), transforms=self._transforms, **{key: value}
            )

        # If no custom function was installed, install the default function:
        if not plan.has_function():
            optimizer = plan.add_step("optimizer")
            tracker = plan.add_handler(
                "tracker",
                constraint_tolerance=self._constraint_tolerance,
                sources={optimizer},
                transforms=self._transforms,
            )
            plan.add_function(_run_func)

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
            if self._transforms is not None:
                results = tuple(
                    item.transform_from_optimizer(self._transforms) for item in results
                )
            callback(results)

        self._observers.append((EventType.FINISHED_EVALUATION, _results_callback))
        return self
