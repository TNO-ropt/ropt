"""This module defines a basic optimization object."""

from __future__ import annotations

import json
import os
import sysconfig
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ropt.config import EnOptConfig
from ropt.enums import EventType, ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.plugins import plugin_manager

from ._factory import create_compute_step, create_evaluator, create_event_handler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import ArrayLike, NDArray

    from ropt.evaluator import EvaluatorCallback
    from ropt.results import FunctionResults
    from ropt.transforms import OptModelTransforms
    from ropt.workflow import Event


@dataclass(slots=True)
class _Results:
    results: FunctionResults | None
    variables: NDArray[np.float64] | None
    exit_code: ExitCode = ExitCode.UNKNOWN


class BasicOptimizer:
    r"""A class for executing single optimization runs.

    The `BasicOptimizer` is designed to simplify the process of setting up and
    executing optimization workflows that consist primarily of a single
    optimization run.

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
    - **Customizable ComputeSteps, Handlers, and Evaluators:** While designed
      for single runs, it allows for the addition of custom compute steps and
      event handlers for more
      complex scenarios.

    By encapsulating the core elements of an optimization run, the
    `BasicOptimizer` reduces the boilerplate code required for simple
    optimization tasks, allowing users to focus on defining the optimization
    problem and analyzing the results.

    The following example demonstrates how to find the optimum of the Rosenbrock
    function using a `BasicOptimizer` object, combining it with a `tracker` to
    store the best result.

    Example:
        ````python
        import numpy as np
        from numpy.typing import NDArray

        from ropt.evaluator import EvaluatorContext, EvaluatorResult
        from ropt.workflow import BasicOptimizer

        DIM = 5
        CONFIG = {
            "variables": {
                "variable_count": DIM,
                "perturbation_magnitudes": 1e-6,
            },
        }
        initial_values = 2 * np.arange(DIM) / DIM + 0.5


        def rosenbrock(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
            objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
            for v_idx in range(variables.shape[0]):
                for d_idx in range(DIM - 1):
                    x, y = variables[v_idx, d_idx : d_idx + 2]
                    objectives[v_idx, 0] += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
            return EvaluatorResult(objectives=objectives)


        optimal_result = BasicOptimizer(CONFIG, rosenbrock).run(initial_values).results

        print(f"variables: {optimal_result.evaluations.variables}")
        print(f"objective: {optimal_result.functions.weighted_objective}")
        ````

    Note: Customization
        The optimization workflow executed by `BasicOptimizer` can be tailored
        in two main ways: by adding event handlers to the default workflow or by
        running an entirely different workflow:

        1.  **Adding Custom Event Handlers**

            This method allows for custom processing of events emitted by the
            *default* optimization workflow, without replacing the workflow
            itself. This is useful for tasks like custom logging or data
            processing.

            Event handlers can be specified in two ways, and handlers from both
            sources will be combined:

            *   **Environment Variable**: If the `ROPT_HANDLERS` environment
                variable contains a comma-separated list of event handler names,
                these handlers will be added to the default optimization
                workflow. Each name must correspond to a registered
                `EventHandler`.

            *   **JSON Configuration File**: If a JSON configuration file is
                found at `<prefix>/share/ropt/options.json` (where `<prefix>` is
                the Python installation prefix or a system-wide data
                prefix.[^1]), `BasicOptimizer` will look for specific keys to
                load additional event handlers. If this JSON file contains a
                `basic_optimizer` key, and nested within it an `event_handlers`
                key, the value of `event_handlers` should be a list of strings.
                Each string in this list should be the name of a registered
                `EventHandler`. These handlers will be added to
                those found via `ROPT_HANDLERS`.

                Example `shared/ropt/options.json`:

                ```json
                {
                    "basic_optimizer": {
                        "event_handlers": ["custom_logger", "extra/event_processor"]
                    }
                }
                ```

            Note that if a custom optimization workflow is installed using the
            `ROPT_SCRIPT` environment variable (see below), these custom
            handlers will not be installed.

        2.  **Custom Workflow Execution**

            If the `ROPT_SCRIPT` environment variable contains an option in the
            format `step-name=script.py` (where `script.py` may be any file),
            the named custom compute step will be executed *instead* of the standard
            optimization workflow, passing it the name of the script that
            defines the new optimization workflow.

            The custom compute step (`step-name`) must adhere to the following:

            *   It must be a registered `ComputeStep`.
            *   Its `run` method  must accept
                1.   An `evaluator` keyword argument, which will receive the
                     evaluator function passed to `BasicOptimizer`.
                2.   A `script` keyword argument, which will receive the name of
                     script passed via `ROPT_SCRIPT`.
            *   This method must return a *callable* that returns an
                optimization [`ExitCode`][ropt.enums.ExitCode].

                This callable will then be executed by `BasicOptimizer` in place
                of its default workflow.

            As a short-cut is possible to also define `ROPT_SCRIPT` with only
            the name of the script (i.e. `ROPT_SCRIPT=script.py`). In this case
            a compute step with the name `run_script` is assumed to exists and
            will be used.

            [^1]:
                The exact path to Python installation prefix, or the system's
                data prefix can be found using the Python `sysconfig` module:
                ```python
                from sysconfig import get_paths
                print(get_paths()["data"])
                ```
    """

    def __init__(
        self,
        enopt_config: dict[str, Any],
        evaluator: EvaluatorCallback,
        *,
        transforms: OptModelTransforms | None = None,
        constraint_tolerance: float = 1e-10,
    ) -> None:
        """Initialize a `BasicOptimizer` object.

        This constructor sets up the necessary components for a single
        optimization run. It requires an optimization configuration, an
        evaluator, and optional domain transform, which together define the
        optimization problem.

        The `constraint_tolerance` is used to check any constraints, if a
        constraint value is within this tolerance, it is considered satisfied.

        Args:
            enopt_config:         The configuration for the optimization.
            evaluator:            The evaluator object.
            transforms:           Optional transforms to apply to the model.
            constraint_tolerance: The constraint violation tolerance.
        """
        self._config = EnOptConfig.model_validate(enopt_config, context=transforms)
        self._transforms = transforms
        self._constraint_tolerance = constraint_tolerance
        self._evaluator = evaluator
        self._observers: list[tuple[EventType, Callable[[Event], None]]] = []
        self._results: _Results

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
    def exit_code(self) -> ExitCode:
        """Return the exit code of the optimization run.

        This property provides access to the [`ExitCode`][ropt.enums.ExitCode]
        that indicates the outcome of the optimization process. It can be used
        to determine whether the optimization completed successfully, was
        aborted, or encountered an error.

        Returns:
            The exit code of the optimization run.
        """
        return self._results.exit_code

    def run(self, initial_values: ArrayLike) -> Self:
        """Run the optimization process.

        This method initiates and executes the optimization workflow defined by
        the `BasicOptimizer` object. It manages the optimization, result
        handling, and event processing. After the optimization is complete, the
        optimal results, variables, and exit code can be accessed via the
        corresponding properties.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """
        # Optionally run a custom compute step defined via the environment:
        custom_function = self._get_custom_compute_step()

        if custom_function is None:
            evaluator = create_evaluator("function_evaluator", callback=self._evaluator)
            tracker = create_event_handler(
                "tracker",
                constraint_tolerance=self._constraint_tolerance,
            )
            optimizer = create_compute_step(
                "optimizer", evaluator=evaluator
            ).add_event_handler(tracker)
            for event_type, function in self._observers:
                optimizer.add_event_handler(
                    create_event_handler(
                        "observer",
                        event_types={event_type},
                        callback=function,
                    )
                )
            for handler in self._custom_event_handlers():
                optimizer.add_event_handler(create_event_handler(handler))

            exit_code = optimizer.run(
                variables=np.asarray(initial_values, dtype=np.float64),
                config=self._config,
                transforms=self._transforms,
            )
            results = tracker["results"]
            variables = None if results is None else results.evaluations.variables
        else:
            exit_code = custom_function()
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
        [`ExitCode.USER_ABORT`][ropt.enums.ExitCode].

        The callback function should have no arguments and return a boolean
        value.

        Args:
            callback: The callable to check for abort conditions.

        Returns:
            The `BasicOptimizer` instance, allowing for method chaining.
        """

        def _check_abort_callback(event: Event) -> None:  # noqa: ARG001
            if callback():
                raise ComputeStepAborted(exit_code=ExitCode.USER_ABORT)

        self._observers.append((EventType.START_EVALUATION, _check_abort_callback))
        return self

    def set_results_callback(self, callback: Callable[..., None]) -> Self:
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
            transforms = event.data["transforms"]
            results = tuple(
                item
                if transforms is None
                else item.transform_from_optimizer(event.data["config"], transforms)
                for item in event.data.get("results", ())
            )
            callback(results)

        self._observers.append((EventType.FINISHED_EVALUATION, _results_callback))
        return self

    def _get_custom_compute_step(self) -> Callable[[], ExitCode] | None:
        if "ROPT_SCRIPT" in os.environ:
            compute_step_name, sep, script = os.environ["ROPT_SCRIPT"].partition("=")
            if not sep:
                compute_step_name, script = "run_script", compute_step_name
            if (
                plugin_manager.get_plugin_name("compute_step", compute_step_name)
                is not None
            ):
                compute_step: Callable[[], ExitCode] = create_compute_step(
                    compute_step_name
                ).run(evaluator=self._evaluator, script=script)
                return compute_step
        return None

    def _custom_event_handlers(self) -> Iterator[str]:
        handlers = os.environ.get("ROPT_HANDLERS", "").split(",")
        handlers += _get_option("event_handlers")
        for handler in dict.fromkeys(handlers):
            if plugin_manager.get_plugin_name("event_handler", handler) is not None:
                yield handler


@cache
def _get_option(option: str) -> list[str]:
    data_path = Path(sysconfig.get_paths()["data"])
    path = data_path / "share" / "ropt" / "options.json"
    with (
        suppress(OSError, json.JSONDecodeError),
        path.open("r", encoding="utf-8") as file_obj,
    ):
        return list(json.load(file_obj).get("basic_optimizer", {}).get(option, []))
    return []
