"""This module defines a basic optimization object."""

from __future__ import annotations

import importlib
import json
import sysconfig
from contextlib import suppress
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.config import EnOptConfig
from ropt.enums import EnOptEventType, ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.workflow.evaluators import Evaluator

from .compute_steps import EnsembleOptimizer
from .event_handlers import Observer, Tracker

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import ArrayLike, NDArray

    from ropt.evaluator import EvaluatorCallback, EvaluatorContext, EvaluatorResult
    from ropt.events import EnOptEvent
    from ropt.results import FunctionResults
    from ropt.transforms import OptModelTransforms


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


        optimizer = BasicOptimizer(CONFIG, rosenbrock)
        optimizer.run(initial_values)

        print(f"Optimal variables: {optimizer.results.evaluations.variables}")
        print(f"Optimal objective: {optimizer.results.functions.weighted_objective}")
        ````

    Note: Customization
        The optimization workflow executed by `BasicOptimizer` can be tailored
        by adding default event handlers. This allows for custom processing of
        events emitted by the *default* optimization workflow, without replacing
        the workflow itself. This is useful for tasks like custom logging or
        data processing.

        Default event handlers can be specified using a JSON configuration file
        is found at `<prefix>/share/ropt/options.json`, where `<prefix>` is the
        Python installation prefix or a system-wide data prefix.[^1]. This JSON
        file should contain a `basic_optimizer` item, containing an
        `event_handlers` item that provides a list of strings of the form
        `"module_name.handler_name"`. The `module_name` denotes a module
        containing an event handler class with the name `module_name`.

        Example `shared/ropt/options.json`:

        ```json
        {
            "basic_optimizer": {
                "event_handlers": ["mylogger.Logger"]
            }
        }
        ```

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
        evaluator: EvaluatorCallback | Evaluator,
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
        self._observers: list[tuple[EnOptEventType, Callable[[EnOptEvent], None]]] = []
        self._results: FunctionResults | None

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
        return self._results

    def run(self, initial_values: ArrayLike) -> ExitCode:
        """Run the optimization process.

        This method initiates and executes the optimization workflow defined by
        the `BasicOptimizer` object. It manages the optimization, result
        handling, and event processing. After the optimization is complete, the
        optimal results, variables, and exit code can be accessed via the
        corresponding properties.

        Returns:
            The exit code returned by the optimization workflow.
        """
        evaluator = (
            self._evaluator
            if isinstance(self._evaluator, Evaluator)
            else _Evaluator(callback=self._evaluator)
        )
        tracker = Tracker(constraint_tolerance=self._constraint_tolerance)
        optimizer = EnsembleOptimizer(evaluator=evaluator)
        optimizer.add_event_handler(tracker)
        for event_type, function in self._observers:
            optimizer.add_event_handler(
                Observer(event_types={event_type}, callback=function)
            )
        for handler in _custom_event_handlers():
            optimizer.add_event_handler(handler())

        exit_code = optimizer.run(
            variables=np.asarray(initial_values, dtype=np.float64),
            config=self._config,
        )
        self._results = tracker["results"]

        return exit_code if isinstance(exit_code, ExitCode) else ExitCode.UNKNOWN

    def set_abort_callback(self, callback: Callable[[], bool]) -> None:
        """Set a callback to check for abort conditions.

        The provided callback function will be invoked repeatedly during the
        optimization process. If the callback returns `True`, the optimization
        will be aborted, and the `BasicOptimizer` will exit with an
        [`ExitCode.USER_ABORT`][ropt.enums.ExitCode].

        The callback function should have no arguments and return a boolean
        value.

        Args:
            callback: The callable to check for abort conditions.
        """

        def _check_abort_callback(event: EnOptEvent) -> None:  # noqa: ARG001
            if callback():
                raise ComputeStepAborted(exit_code=ExitCode.USER_ABORT)

        self._observers.append((EnOptEventType.START_EVALUATION, _check_abort_callback))

    def set_results_callback(self, callback: Callable[..., None]) -> None:
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
        """

        def _results_callback(event: EnOptEvent) -> None:
            results = tuple(
                item
                if event.config.transforms is None
                else item.transform_from_optimizer(event.config)
                for item in event.results
            )
            callback(results)

        self._observers.append((EnOptEventType.FINISHED_EVALUATION, _results_callback))


class _Evaluator(Evaluator):
    def __init__(self, *, callback: EvaluatorCallback) -> None:
        super().__init__()
        self._evaluator_callback = callback

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        return self._evaluator_callback(variables, context)


def _custom_event_handlers() -> Iterator[Any]:
    handlers = _get_option("event_handlers")
    for handler in handlers:
        module_path, class_name = handler.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            continue
        yield getattr(module, class_name)


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
