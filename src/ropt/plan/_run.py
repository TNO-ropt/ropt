"""This module defines a basic optimization object."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path  # noqa: TCH003
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

from ropt.config.plan import PlanConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.plugins import PluginManager
from ropt.report import ResultsTable
from ropt.results import convert_to_maximize

from ._plan import OptimizerContext, Plan

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.evaluator import Evaluator
    from ropt.optimization import Event
    from ropt.plugins._manager import PluginType
    from ropt.plugins.base import Plugin
    from ropt.results import FunctionResults

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_RepeatTypes = Literal["initial", "last", "optimal", "last_optimal"]


@dataclass
class _Repeat:
    iterations: int
    restart_from: _RepeatTypes
    metadata_var: Optional[str]


@dataclass
class _Results:
    results: Optional[FunctionResults]
    variables: Optional[NDArray[np.float64]]
    exit_code: OptimizerExitCode = OptimizerExitCode.UNKNOWN


class OptimizationPlanRunner:
    """A class for running optimization plans.

    `OptimizationPlanRunner` objects are designed for use cases where the
    optimization workflow is relatively simple, such as a single optimization
    run possibly with a few restarts. Using this object can be more convenient
    than defining and running an optimization plan directly in such cases.

    This class provides the following features:

    - Start a single optimization.
    - Repeat the same optimization multiple times, with various options for
      restarting from different points.
    - Add observer functions connected to various optimization events.
    - Attach metadata to each result generated during the optimization.
    - Generate tables summarizing the optimization results.
    """

    def __init__(
        self,
        enopt_config: Union[Dict[str, Any], EnOptConfig],
        evaluator: Evaluator,
        *,
        constraint_tolerance: float = 1e-10,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize an `OptimizationPlanRunner` object.

        An optimization configuration and an evaluation object must be provided,
        as they define the optimization to perform.

        Args:
            enopt_config:         The configuration for the optimization.
            evaluator:            The evaluator object used to evaluate functions.
            constraint_tolerance: The tolerance level used to detect constraint violations.
            seed:                 The seed for the random number generator used
                                  for stochastic gradient estimation.
        """
        self._plugin_manager: Optional[PluginManager] = None
        self._optimizer_context = OptimizerContext(evaluator=evaluator, seed=seed)
        self._observers: List[Tuple[EventType, Callable[[Event], None]]] = []
        self._metadata: Dict[str, Any] = {}
        self._repeat: Optional[_Repeat] = None
        self._plan_config: Dict[str, Any] = {
            "context": [
                {
                    "id": "__config__",
                    "init": "config",
                    "with": enopt_config,
                },
                {
                    "id": "__optimum_tracker__",
                    "init": "tracker",
                    "with": {"constraint_tolerance": constraint_tolerance},
                },
            ],
            "steps": [
                {
                    "name": "__optimizer_step__",
                    "run": "optimizer",
                    "with": {
                        "config": "$__config__",
                        "update": ["__optimum_tracker__"],
                        "exit_code_var": "exit_code",
                    },
                }
            ],
        }
        self._results: _Results

    @property
    def results(self) -> Optional[FunctionResults]:
        """Return the optimal result.

        Returns:
            The optimal result found during optimization.
        """
        return self._results.results

    @property
    def variables(self) -> Optional[NDArray[np.float64]]:
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

    def add_plugins(self, plugin_type: PluginType, plugins: Dict[str, Plugin]) -> Self:
        """Add plugins.

        By default, plugins are installed via Python's entry point mechanism.
        This method allows you to install additional plugins.

        Args:
            plugin_type: The type of plugin to install.
            plugins:     A dictionary mapping plugin names to plugin objects.

        Returns:
            The `OptimizationPlanRunner` instance, allowing for method chaining.
        """
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager()
        self._plugin_manager.add_plugins(plugin_type, plugins)
        return self

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
            The `OptimizationPlanRunner` instance, allowing for method chaining.
        """
        self._observers.append((event_type, function))
        return self

    def add_metadata(self, metadata: Dict[str, Any]) -> Self:
        """Add metadata.

        Add a dictionary of metadata that will be attached to each result object
        generated during optimization.

        Args:
            metadata: The dictionary containing metadata to add to each result.

        Returns:
            The `OptimizationPlanRunner` instance, allowing for method chaining.
        """
        for key, value in metadata.items():
            if value is None:
                del self._metadata[key]
            else:
                self._metadata[key] = value
        return self

    def _handle_results(
        self, event: Event, table: ResultsTable, *, maximize: bool
    ) -> None:
        assert event.results is not None
        table.add_results(
            event.config,
            (
                (convert_to_maximize(item) for item in event.results)
                if maximize
                else event.results
            ),
        )

    def add_table(
        self,
        columns: Dict[str, str],
        path: Path,
        table_type: Literal["functions", "gradients"] = "functions",
        min_header_len: Optional[int] = None,
        *,
        maximize: bool = False,
    ) -> Self:
        """Add a table of results.

        This method instructs the runner to generate a table summarizing the
        results of the optimization. This is implemented via a
        [`ResultsTable`][ropt.report.ResultsTable] object. Refer to its
        documentation for more details.

        Args:
            columns:        A mapping of column names for the results table.
            path:           The location where the results file will be saved.
            table_type:     The type of table to generate.
            min_header_len: The minimum number of header lines to generate.
            maximize:       If `True`, interpret the results as a maximization
                            problem rather than the default minimization.

        Returns:
            The `OptimizationPlanRunner` instance, allowing for method chaining.
        """
        table = ResultsTable(
            columns=columns,
            path=path,
            table_type=table_type,
            min_header_len=min_header_len,
        )
        self._observers.append(
            (
                EventType.FINISHED_EVALUATION,
                partial(self._handle_results, table=table, maximize=maximize),
            ),
        )
        return self

    def repeat(
        self,
        iterations: int,
        restart_from: _RepeatTypes = "optimal",
        metadata_var: Optional[str] = None,
    ) -> Self:
        """Repeat the optimization.

        Run the optimization multiple times with various options for the
        starting points. On the first run, the optimization starts from the
        initial variables defined in its configuration. For subsequent runs, the
        initial variables are selected based on the `restart_from` option:

        - `"initial"`: Use the initial values from the configuration.
        - `"last"`: Use the variables from the previous run.
        - `"optimal"`: Use the variables from the optimal result found so far.
        - `"last_optimal"`: Use the variables from the optimal result of the last run.

        If `metadata_var` is defined, a field will be added to the metadata
        stored with each result, recording the sequence number of the
        optimization run.

        Args:
            iterations:      The number of times to run the optimization.
            restart_from:    The method for selecting initial variables. Defaults to `"optimal"`.
            metadata_var:    Optional field name in the metadata to record the repeat index.

        Returns:
            The `OptimizationPlanRunner` instance, allowing for method chaining.
        """
        if self._repeat is not None:
            msg = "The repeat() method can only be called once."
            raise RuntimeError(msg)
        self._repeat = _Repeat(
            iterations=iterations,
            restart_from=restart_from,
            metadata_var=metadata_var,
        )
        return self

    def _build_plan_config(self) -> Dict[str, Any]:
        context = self._plan_config["context"]
        steps = self._plan_config["steps"]
        metadata = self._metadata

        if self._repeat is not None:
            if self._repeat.metadata_var is not None:
                metadata[self._repeat.metadata_var] = "$__repeat_counter__"
            context, steps = _add_repeat_tracker(
                context, steps, self._repeat.restart_from
            )

        if metadata:
            steps = [
                {
                    "run": "metadata",
                    "with": {
                        "metadata": self._metadata,
                    },
                },
                *steps,
            ]

        if self._repeat is not None:
            steps = [
                {
                    "run": "repeat",
                    "with": {
                        "iterations": self._repeat.iterations,
                        "counter_var": "__repeat_counter__",
                        "steps": steps,
                    },
                }
            ]

        return {"context": context, "steps": steps}

    def run(self) -> Self:
        """Run the optimization.

        Returns:
            The `OptimizationPlanRunner` instance, allowing for method chaining.
        """
        plan = Plan(
            PlanConfig.model_validate(self._build_plan_config()),
            self._optimizer_context,
            plugin_manager=self._plugin_manager,
        )
        for event_type, function in self._observers:
            plan.optimizer_context.events.add_observer(event_type, function)
        plan.run()
        results = plan["__optimum_tracker__"]
        self._results = _Results(
            results=results,
            variables=None if results is None else results.evaluations.variables,
            exit_code=plan["exit_code"],
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


def _add_repeat_tracker(
    context: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    restart_from: Literal["initial", "last", "optimal", "last_optimal"] = "optimal",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    idx = next(idx for idx, step in enumerate(steps) if step["run"] == "optimizer")
    if restart_from in ["last", "last_optimal"]:
        context.append(
            {
                "id": "__repeat_tracker__",
                "init": "tracker",
                "with": {"type": "last" if restart_from == "last" else "optimal"},
            }
        )
        steps[idx]["with"]["update"].append("__repeat_tracker__")
    if restart_from == "last":
        steps[idx]["with"]["initial_values"] = "$__repeat_tracker__"
    elif restart_from == "optimal":
        steps[idx]["with"]["initial_values"] = "$__optimum_tracker__"
    elif restart_from == "last_optimal":
        steps[idx]["with"]["initial_values"] = "$__initial_var__"
        steps = [
            {"run": "setvar", "with": "__initial_var__ = $__repeat_tracker__"},
            {"run": "reset", "with": {"context": "__repeat_tracker__"}},
            *steps,
        ]
    return context, steps
