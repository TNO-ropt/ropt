"""This module defines a basic optimization object."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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

from ._plan import OptimizerContext, Plan

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.evaluator import Evaluator
    from ropt.plan import Event
    from ropt.plugins._manager import PluginType
    from ropt.plugins.base import Plugin
    from ropt.results import FunctionResults

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass
class _Results:
    results: Optional[FunctionResults]
    variables: Optional[NDArray[np.float64]]
    exit_code: OptimizerExitCode = OptimizerExitCode.UNKNOWN


class OptimizationPlanRunner:
    """A class for running optimization plans.

    `OptimizationPlanRunner` objects are designed for use cases where the
    optimization workflow comprises a single optimization run. Using this object
    can be more convenient than defining and running an optimization plan
    directly in such cases.

    This class provides the following features:

    - Start a single optimization.
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
        self._variables = {
            "__config__": enopt_config,
            "__optimum_tracker__": None,
            "__exit_code__": OptimizerExitCode.UNKNOWN,
        }
        self._steps: List[Dict[str, Any]] = [
            {
                "run": "optimizer",
                "with": {
                    "config": "$__config__",
                    "exit_code_var": "__exit_code__",
                    "tags": "__optimizer_tag__",
                },
            }
        ]
        self._handlers: List[Dict[str, Any]] = [
            {
                "run": "tracker",
                "with": {
                    "var": "__optimum_tracker__",
                    "constraint_tolerance": constraint_tolerance,
                    "tags": "__optimizer_tag__",
                },
            },
        ]
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
        self._handlers.append(
            {
                "run": "table",
                "with": {
                    "tags": "__optimizer_tag__",
                    "columns": columns,
                    "path": path,
                    "table_type": table_type,
                    "min_header_len": min_header_len,
                    "maximize": maximize,
                },
            }
        )
        return self

    def run(self) -> Self:
        """Run the optimization.

        Returns:
            The `OptimizationPlanRunner` instance, allowing for method chaining.
        """
        plan = Plan(
            PlanConfig.model_validate(
                {
                    "variables": self._variables,
                    "steps": self._steps,
                    "results": self._handlers,
                }
            ),
            self._optimizer_context,
            plugin_manager=self._plugin_manager,
        )
        for event_type, function in self._observers:
            plan.add_observer(event_type, function)
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
