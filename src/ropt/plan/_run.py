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
from uuid import uuid4

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
    counter_var: Optional[str]
    metadata_var: Optional[str]


@dataclass
class _Results:
    results: Optional[FunctionResults]
    variables: Optional[NDArray[np.float64]]
    exit_code: OptimizerExitCode = OptimizerExitCode.UNKNOWN


class OptimizationPlanRunner:
    """A class for running optimization plans."""

    def __init__(
        self,
        enopt_config: Union[Dict[str, Any], EnOptConfig],
        evaluator: Evaluator,
        *,
        constraint_tolerance: float = 1e-10,
        seed: Optional[int] = None,
    ) -> None:
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
        return self._results.results

    @property
    def variables(self) -> Optional[NDArray[np.float64]]:
        return self._results.variables

    @property
    def exit_code(self) -> OptimizerExitCode:
        return self._results.exit_code

    def add_plugins(self, plugin_type: PluginType, plugins: Dict[str, Plugin]) -> Self:
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager()
        self._plugin_manager.add_plugins(plugin_type, plugins)
        return self

    def add_observer(
        self, event_type: EventType, function: Callable[[Event], None]
    ) -> Self:
        self._observers.append((event_type, function))
        return self

    def add_metadata(self, metadata: Dict[str, Any]) -> Self:
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
        self._plan_config["context"].append(
            {
                "id": f"__{uuid4().hex}__",
                "init": "table",
                "with": {
                    "columns": columns,
                    "path": path,
                    "table_type": table_type,
                    "min_header_len": min_header_len,
                    "maximize": maximize,
                    "steps": ["__optimizer_step__"],
                },
            }
        )
        return self

    def repeat(
        self,
        iterations: int,
        restart_from: _RepeatTypes = "optimal",
        counter_var: Optional[str] = None,
        metadata_var: Optional[str] = None,
    ) -> Self:
        if self._repeat is not None:
            msg = "The repeat() method can only be called once."
            raise RuntimeError(msg)
        self._repeat = _Repeat(
            iterations=iterations,
            restart_from=restart_from,
            counter_var=counter_var,
            metadata_var=metadata_var,
        )
        return self

    def _build_plan_config(self) -> Dict[str, Any]:
        context = self._plan_config["context"]
        steps = self._plan_config["steps"]
        metadata = self._metadata

        if self._repeat is not None:
            counter_var = self._repeat.counter_var
            if self._repeat.metadata_var is not None:
                if counter_var is None:
                    counter_var = "__repeat_counter__"
                metadata[self._repeat.metadata_var] = f"${counter_var}"
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
                        "counter_var": counter_var,
                        "steps": steps,
                    },
                }
            ]

        return {"context": context, "steps": steps}

    def run(self) -> Self:
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
