"""This module defines a basic optimization object."""

from __future__ import annotations

import sys
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
    from ropt.events import Event
    from ropt.plugins._manager import PluginType
    from ropt.plugins.base import Plugin
    from ropt.results import FunctionResults

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


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
        self._enopt_config = enopt_config
        self._evaluator = evaluator
        self._plugin_manager: Optional[PluginManager] = None
        self._context = OptimizerContext(evaluator=self._evaluator, seed=seed)
        self._results: Optional[FunctionResults]
        self._variables: Optional[NDArray[np.float64]]
        self._exit_code: OptimizerExitCode = OptimizerExitCode.UNKNOWN
        self._observers: List[Tuple[EventType, Callable[[Event], None]]] = []

        self._plan_config: Dict[str, List[Dict[str, Any]]] = {
            "context": [
                {
                    "id": "config",
                    "init": "config",
                    "with": enopt_config,
                },
                {
                    "id": "optimal",
                    "init": "tracker",
                    "with": {"constraint_tolerance": constraint_tolerance},
                },
            ],
            "steps": [
                {
                    "run": "optimizer",
                    "with": {
                        "config": "$config",
                        "update": ["optimal"],
                        "exit_code_var": "exit_code",
                    },
                },
            ],
        }

    @property
    def results(self) -> Optional[FunctionResults]:
        return self._results

    @property
    def variables(self) -> Optional[NDArray[np.float64]]:
        return self._variables

    @property
    def exit_code(self) -> OptimizerExitCode:
        return self._exit_code

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
        steps = self._plan_config["steps"]
        idx = next(
            (idx for idx, step in enumerate(steps) if step["run"] == "repeat"), None
        )
        if idx is not None:
            steps = steps[idx]["with"]["steps"]
        idx = next(idx for idx, step in enumerate(steps) if step["run"] == "optimizer")
        steps[idx]["with"]["metadata"] = metadata
        return self

    def repeat(
        self,
        iterations: int,
        restart_from: Literal["initial", "last", "optimal", "last_optimal"] = "optimal",
        counter_var: Optional[str] = None,
    ) -> Self:
        if any(step["run"] == "repeat" for step in self._plan_config["steps"]):
            msg = "The repeat() method can only be called once."
            raise RuntimeError(msg)
        steps = self._plan_config["steps"]
        idx = next(idx for idx, step in enumerate(steps) if step["run"] == "optimizer")
        if restart_from in ["last", "last_optimal"]:
            self._plan_config["context"].append(
                {
                    "id": "repeat_tracker",
                    "init": "tracker",
                    "with": {"type": "last" if restart_from == "last" else "optimal"},
                }
            )
            steps[idx]["with"]["update"].append("repeat_tracker")
        if restart_from == "last":
            steps[idx]["with"]["initial_values"] = "$repeat_tracker"
        elif restart_from == "optimal":
            steps[idx]["with"]["initial_values"] = "$optimal"
        elif restart_from == "last_optimal":
            steps[idx]["with"]["initial_values"] = "$initial"
            steps = [
                {"run": "setvar", "with": "initial = $repeat_tracker"},
                {"run": "reset", "with": {"context": "repeat_tracker"}},
                *steps,
            ]
        self._plan_config["steps"] = [
            {
                "run": "repeat",
                "with": {
                    "counter_var": counter_var,
                    "iterations": iterations,
                    "steps": steps,
                },
            }
        ]
        return self

    def run(self) -> Self:
        config = PlanConfig.model_validate(self._plan_config)
        plan = Plan(
            config,
            self._context,
            plugin_manager=self._plugin_manager,
        )
        for event_type, function in self._observers:
            plan.optimizer_context.events.add_observer(event_type, function)
        plan.run()
        self._results = plan["optimal"]
        self._variables = (
            None if self._results is None else self._results.evaluations.variables
        )
        self._exit_code = plan["exit_code"]
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
