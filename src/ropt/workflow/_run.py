"""This module defines workflow object."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

from ropt.config.workflow import WorkflowConfig
from ropt.enums import OptimizerExitCode
from ropt.exceptions import OptimizationAborted

from ._workflow import OptimizerContext, Workflow

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.evaluator import Evaluator
    from ropt.plugins import PluginManager
    from ropt.results import FunctionResults, Results


def _basic_run_config(
    enopt_config: Union[Dict[str, Any], EnOptConfig],
    callback: Optional[Callable[[Tuple[Results, ...]], None]],
    constraint_tolerance: float,
) -> Dict[str, Any]:
    updates = ["optimal"]
    if callback is not None:
        updates.append("callback")
    config: Dict[str, Any] = {
        "context": [
            {
                "id": "config",
                "init": "config",
                "with": enopt_config,
            },
            {
                "id": "optimal",
                "init": "results",
                "with": {"constraint_tolerance": constraint_tolerance},
            },
        ],
        "steps": [
            {
                "run": "optimizer",
                "with": {
                    "config": "$config",
                    "update": updates,
                    "exit_code": "exit_code",
                },
            },
        ],
    }
    if callback is not None:
        config["context"].append(
            {
                "id": "callback",
                "init": "callback",
                "with": {"function": callback},
            }
        )
    return config


class BasicWorkflow:
    """Runner class for basic workflows."""

    def __init__(  # noqa: PLR0913
        self,
        enopt_config: Union[Dict[str, Any], EnOptConfig],
        evaluator: Evaluator,
        *,
        constraint_tolerance: float = 1e-10,
        callback: Optional[Callable[[Tuple[Results, ...]], None]] = None,
        seed: Optional[int] = None,
        plugin_manager: Optional[PluginManager] = None,
    ) -> None:
        self._enopt_config = enopt_config
        self._evaluator = evaluator
        self._plugin_manager = plugin_manager
        self._context = OptimizerContext(evaluator=self._evaluator, seed=seed)
        self._results: Optional[FunctionResults]
        self._variables: Optional[NDArray[np.float64]]
        self._exit_code: OptimizerExitCode = OptimizerExitCode.UNKNOWN

        self.workflow_config = _basic_run_config(
            self._enopt_config, callback, constraint_tolerance
        )

    @property
    def results(self) -> Optional[FunctionResults]:
        return self._results

    @property
    def variables(self) -> Optional[NDArray[np.float64]]:
        return self._variables

    @property
    def exit_code(self) -> OptimizerExitCode:
        return self._exit_code

    def run(self) -> BasicWorkflow:
        config = WorkflowConfig.model_validate(self.workflow_config)
        workflow = Workflow(
            config,
            self._context,
            plugin_manager=self._plugin_manager,
        )
        workflow.run()
        self._results = workflow["optimal"]
        self._variables = (
            None if self._results is None else self._results.evaluations.variables
        )
        self._exit_code = workflow["exit_code"]
        return self

    @staticmethod
    def abort_optimization() -> NoReturn:
        """Abort the current optimization run.

        This method can be called from within callbacks to interrupt the ongoing
        optimization plan. The exact point at which the optimization is aborted
        depends on the step in the plan that is executing at that point. For
        example, within a running optimizer, the process will be interrupted
        after completing the current function evaluation.
        """
        raise OptimizationAborted(exit_code=OptimizerExitCode.USER_ABORT)
