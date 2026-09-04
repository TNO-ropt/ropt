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

from ropt.components.compute_steps import OptimizationStep
from ropt.components.evaluators import BatchEvaluator, Evaluator
from ropt.components.event_handlers import CallbackHandler, ResultsHandler
from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, ExitCode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import ArrayLike

    from ropt.evaluation import EvaluationBatchCallback
    from ropt.events import EnOptEvent
    from ropt.results import FunctionResults


class BasicOptimizer:
    r"""A simple interface for single optimization runs.

    Wraps the workflow components into a run-once interface with built-in result
    tracking. Passing a configuration dictionary and an evaluator is enough to
    run an optimization and retrieve the best result. Internally it:

    1. validates `config` into an [`EnOptContext`][ropt.context.EnOptContext];
    2. wraps a plain
       [`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback] in a
       [`BatchEvaluator`][ropt.components.evaluators.BatchEvaluator], or uses a
       supplied [`Evaluator`][ropt.components.evaluators.Evaluator] as given;
    3. creates an
       [`OptimizationStep`][ropt.components.compute_steps.OptimizationStep] and
       attaches a
       [`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] to
       remember the best result;
    4. runs the step, exposing the best
       [`FunctionResults`][ropt.results.FunctionResults] via
       [`results`][ropt.workflow.BasicOptimizer.results].

    Progress can be monitored by registering a callback with
    [`set_results_callback`][ropt.workflow.BasicOptimizer.set_results_callback].
    For more control (multiple runs, custom event handlers, or parallel/async
    evaluation) use the workflow components directly.

    **Injecting event handlers into every run.** Extra event handlers can be
    added to *every* `BasicOptimizer` run without changing any call site, for
    example to add logging, telemetry, or a custom results store. On start-up
    `BasicOptimizer` reads a JSON file at `<prefix>/share/ropt/options.json`,
    where `<prefix>` is the Python installation's data prefix (the value of
    `sysconfig.get_paths()["data"]`). Handlers are listed under
    `basic_optimizer.event_handlers` as `module.ClassName` strings:

    ```json
    {
        "basic_optimizer": {
            "event_handlers": ["mypackage.MyHandler"]
        }
    }
    ```

    Each referenced class must be importable from the active environment and must
    subclass [`EventHandler`][ropt.components.event_handlers.EventHandler] with no
    required constructor arguments; it is instantiated and attached to every run.
    Entries whose module cannot be imported are skipped, and a missing or
    malformed file is ignored.
    """

    def __init__(
        self,
        config: dict[str, Any],
        evaluator: EvaluationBatchCallback | Evaluator,
        *,
        constraint_tolerance: float = 1e-10,
    ) -> None:
        """Initialize a `BasicOptimizer` object.

        Args:
            config:               The configuration for the optimization.
            evaluator:            An
                [`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback]
                callable that evaluates a batch of variable vectors, or an
                [`Evaluator`][ropt.components.evaluators.Evaluator] instance for
                advanced features such as caching, parallel, or HPC evaluation.
            constraint_tolerance: The constraint violation tolerance; a
                constraint within this tolerance is considered satisfied.
        """
        self._context = EnOptContext.model_validate(config)
        self._constraint_tolerance = constraint_tolerance
        self._evaluator = evaluator
        self._observers: list[tuple[EnOptEventType, Callable[[EnOptEvent], None]]] = []
        self._results: FunctionResults | None

    @property
    def results(self) -> FunctionResults | None:
        """The optimal result found during the optimization.

        Returns:
            The optimal result, or `None` if none was found yet.
        """
        return self._results

    def run(self, initial_values: ArrayLike) -> ExitCode:
        """Run the optimization process.

        Args:
            initial_values: The variable vector to start the optimization from.

        Returns:
            The exit code returned by the optimization workflow.
        """
        evaluator = (
            self._evaluator
            if isinstance(self._evaluator, Evaluator)
            else BatchEvaluator(callback=self._evaluator)
        )
        result_handler = ResultsHandler(constraint_tolerance=self._constraint_tolerance)
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
        return exit_code

    def set_results_callback(self, callback: Callable[..., None]) -> None:
        """Set a callback to report new results.

        Invoked with a `tuple[FunctionResults, ...]` whenever new results
        become available:

        ```python
        def callback(results: tuple[FunctionResults, ...]) -> None:
            ...
        ```

        Args:
            callback: The callable that will be invoked to report new results.
        """

        def _results_callback(event: EnOptEvent) -> None:
            results = tuple(item.unscale(event.context) for item in event.results)
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
