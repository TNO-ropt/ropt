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

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, ExitCode
from ropt.exceptions import Abort
from ropt.workflow.evaluators import BatchEvaluator, Evaluator

from .compute_steps import OptimizationStep
from .event_handlers import CallbackHandler, ResultHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import ArrayLike

    from ropt.evaluation import EvaluationBatchCallback
    from ropt.events import EnOptEvent
    from ropt.results import FunctionResults


class BasicOptimizer:
    r"""A simple interface for single optimization runs.

    Wraps the workflow framework into a run-once interface with built-in
    result tracking.

    See [Basic Optimization](../usage/basic.md) for a walkthrough and full
    example.
    """

    def __init__(
        self,
        config: dict[str, Any],
        evaluator: EvaluationBatchCallback | Evaluator,
        *,
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
            config:               The configuration for the optimization.
            evaluator:            The evaluator object.
            constraint_tolerance: The constraint violation tolerance.
        """
        self._context = EnOptContext.model_validate(config)
        self._constraint_tolerance = constraint_tolerance
        self._evaluator = evaluator
        self._observers: list[tuple[EnOptEventType, Callable[[EnOptEvent], None]]] = []
        self._results: FunctionResults | None

    @property
    def results(self) -> FunctionResults | None:
        """The optimal result found during the optimization.

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
            else BatchEvaluator(callback=self._evaluator)
        )
        result_handler = ResultHandler(constraint_tolerance=self._constraint_tolerance)
        optimizer = OptimizationStep(evaluator=evaluator)
        optimizer.add_event_handler(result_handler)
        for event_type, function in self._observers:
            optimizer.add_event_handler(
                CallbackHandler(event_types={event_type}, callback=function)
            )
        for handler in _custom_event_handlers():
            optimizer.add_event_handler(handler())

        exit_code = optimizer.run(
            variables=np.asarray(initial_values, dtype=np.float64),
            context=self._context,
        )
        self._results = result_handler["results"]

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
                raise Abort(exit_code=ExitCode.USER_ABORT)

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
                item.transform_from_optimizer(event.context) for item in event.results
            )
            callback(results)

        self._observers.append((EnOptEventType.FINISHED_EVALUATION, _results_callback))


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
