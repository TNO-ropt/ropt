"""A class for running optimizations with script-based jobs."""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from shutil import rmtree
from string import Template
from traceback import format_exception
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    TextIO,
    Tuple,
    Union,
    no_type_check,
)

import numpy as np
from numpy.typing import NDArray
from parsl.app.app import bash_app
from parsl.providers.base import ExecutionProvider
from tabulate import tabulate

from ropt.config.plan import PlanConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.evaluator import EvaluatorContext
from ropt.evaluator.parsl import ParslEvaluator, State
from ropt.exceptions import ConfigError
from ropt.plan import Event, OptimizerContext, Plan

from ._config import ScriptEvaluatorConfig, ScriptOptimizerConfig
from ._task import ScriptTask
from ._utils import _format_list, _get_function_files, _make_dict

if TYPE_CHECKING:
    from ropt.results import FunctionResults

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@no_type_check
@bash_app()
def run_script(
    _: Any,  # noqa: ANN401
    script: str,
    stdout: TextIO,  # noqa: ARG001
    stderr: TextIO,  # noqa: ARG001
) -> str:
    return script


class ScriptOptimizer:
    """Optimizer class for running script-based optimization plans."""

    def __init__(
        self,
        config: Union[Dict[str, Any], ScriptOptimizerConfig],
        plan: Dict[str, Any],
        tasks: Dict[str, str],
        *,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the optimizer.

        The directory where the jobs run is constructed from by nesting
        directories according the `job_labels` tuple. These directories are
        located in the directory where the optimizer runs, which can be set
        using the `work_dir` parameter of the `run()` method.

        Args:
            config: Script optimizer configuration
            plan:   The optimization plan to run
            tasks:  A dictionary mapping task names to strings containing bash code
            seed:   Optional seed used by the optimization code
        """
        self._config = (
            config
            if isinstance(config, ScriptOptimizerConfig)
            else ScriptOptimizerConfig.model_validate(config)
        )
        self._plan_config = plan
        self._tasks = tasks
        self._seed = seed
        self._status: Dict[int, Any] = {}
        self._optimal_result: Optional[FunctionResults] = None
        self._observers: List[Tuple[EventType, Callable[[Event], None]]] = []

    def _set_logger(self) -> None:
        self._logger = logging.getLogger("ScriptBasedOptimizer")
        self._logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("optimizer.log")
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def _function(
        self,
        batch_id: int,
        job_idx: int,
        variables: NDArray[np.float64],
        context: EvaluatorContext,
    ) -> List[ScriptTask]:
        assert context.config.realizations.names is not None
        realization = context.config.realizations.names[context.realizations[job_idx]]
        if not isinstance(realization, int):
            msg = f"Realization name must be an integer: {realization}"
            raise ConfigError(msg)

        job_labels = tuple(
            label.format(batch=batch_id, realization=realization, job=job_idx)
            for label in self._config.job_labels
        )
        assert self._config.job_dir is not None
        path = self._config.job_dir / Path(*job_labels)
        objective_names, constraint_names = _get_function_files(context.config)
        objective_paths = tuple(path / name for name in objective_names)
        constraint_paths = tuple(path / name for name in constraint_names)

        var_path = path / "var-vector.npy"
        if all(
            path.exists() for path in (var_path, *objective_paths, *constraint_paths)
        ) and np.allclose(np.load(var_path), variables):
            msg = f"Batch {batch_id}, realization {realization}, job {job_idx}: SKIPPED"
            self._logger.info(msg)
            return [
                ScriptTask(
                    future=None,
                    objective_paths=objective_paths,
                    constraint_paths=constraint_paths,
                )
            ]

        if not self._tasks.items():
            return []

        rmtree(path, ignore_errors=True)
        Path.mkdir(path, parents=True)
        np.save(var_path, variables)

        assert context.config.variables.names is not None
        filename = path / self._config.var_filename.with_suffix(".json")
        var_dict = _make_dict(variables, context.config.variables.names)
        with filename.open("w", encoding="utf-8") as file_obj:
            json.dump(var_dict, file_obj, indent=2)

        tasks: List[ScriptTask] = []
        for task_name, script in self._tasks.items():
            substituted_script = Template(script).safe_substitute(
                work_dir=self._config.work_dir, realization=realization
            )
            tasks.append(
                ScriptTask(
                    name=task_name,
                    future=run_script(
                        tasks[-1].future if tasks else None,
                        f"cd {path}\n{substituted_script}",
                        stdout=((path / task_name).with_suffix(".stdout"), "a"),
                        stderr=((path / task_name).with_suffix(".stderr"), "a"),
                    ),
                    job_labels=job_labels,
                    realization=realization,
                ),
            )
        tasks[-1].objective_paths = objective_paths
        tasks[-1].constraint_paths = constraint_paths

        return tasks

    def _log_task(self, task: ScriptTask) -> None:
        if not task.logged:
            job_label = ", ".join(task.job_labels)

            if task.state == State.FAILED:
                assert task.exception is not None
                msg = f"{job_label}, {task.name}: FAILED\n"
                msg += "".join(
                    format_exception(
                        type(task.exception),
                        task.exception,
                        task.exception.__traceback__,
                    ),
                )
                self._logger.error(msg)
                task.logged = True

            if task.state == State.SUCCESS:
                msg = f"{job_label}, {task.name}: FINISHED"
                self._logger.info(msg)
                task.logged = True

    def _write_state_report(
        self, batch_id: int, jobs: Dict[int, List[ScriptTask]]
    ) -> None:
        states: DefaultDict[str, Dict[int, State]] = defaultdict(dict)
        for job_idx, tasks in jobs.items():
            for task in tasks:
                states[task.name][job_idx] = task.state
        table: List[Dict[str, Any]] = []
        for name, task_states in states.items():
            job_name = name
            for state in State:
                job_indices = [
                    idx for idx, item in task_states.items() if item == state
                ]
                if job_indices:
                    table.append(
                        {
                            "Task": job_name,
                            "State": state.value,
                            "Jobs": _format_list(job_indices),
                        },
                    )
                    job_name = ""
        with (self._config.work_dir / "status.txt").open(
            "w", encoding="utf-8"
        ) as file_obj:
            file_obj.write(f"Batch: {batch_id}\n\n")
            table_str = tabulate(table, headers="keys", tablefmt="simple")
            file_obj.write(f"{table_str}\n")

    def _update_current_state(
        self, batch_id: int, jobs: Dict[int, List[ScriptTask]]
    ) -> None:
        # Update the current batch
        states: DefaultDict[int, Dict[str, Any]] = defaultdict(dict)
        for job_idx, tasks in jobs.items():
            for task in tasks:
                states[job_idx][task.name] = {
                    "state": task.state.value,
                    "realization": task.realization,
                }
                self._log_task(task)
        self._status[batch_id] = states

    def _store_states(self) -> None:
        states_json = []
        for batch_id in sorted(self._status.keys()):
            states = self._status[batch_id]
            states_json.append(
                {
                    "batch_id": batch_id,
                    "jobs": [
                        {
                            "job": job_idx,
                            "tasks": [
                                {
                                    "name": task_name,
                                    "state": task_state["state"],
                                    "realization": task_state["realization"],
                                }
                                for task_name, task_state in states[job_idx].items()
                            ],
                        }
                        for job_idx in sorted(self._status[batch_id])
                    ],
                },
            )
        with (self._config.work_dir / "states.json").open(
            "w", encoding="utf-8"
        ) as file_obj:
            json.dump(states_json, file_obj, sort_keys=True, indent=4)

    def _monitor(self, batch_id: int, jobs: Dict[int, List[ScriptTask]]) -> None:
        self._write_state_report(batch_id, jobs)
        self._update_current_state(batch_id, jobs)
        self._store_states()

    def _log_exit_code(self, event: Event) -> None:
        msg = ""
        if event.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS:
            msg = "Too few successful realizations: optimization stopped."
        elif event.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED:
            msg = "Maximum number of functions reached: optimization stopped."
        elif event.exit_code == OptimizerExitCode.USER_ABORT:
            msg = "Optimization plan aborted by the user."
        elif event.exit_code == OptimizerExitCode.OPTIMIZER_STEP_FINISHED:
            msg = "Optimization finished normally."
        if msg:
            if event.tag is not None:
                msg = f"Step tagged as `{event.tag}`: {msg}"
            self._logger.info(msg)

    def add_observer(
        self, event_type: EventType, function: Callable[[Event], None]
    ) -> Self:
        self._observers.append((event_type, function))
        return self

    def run(
        self,
        provider: Optional[ExecutionProvider] = None,
        evaluator_config: Optional[Union[Dict[str, Any], ScriptEvaluatorConfig]] = None,
    ) -> Plan:
        """Run the optimization."""
        cwd = Path.cwd()
        Path.mkdir(self._config.work_dir, parents=True, exist_ok=True)
        os.chdir(self._config.work_dir)

        if evaluator_config is None:
            evaluator_config = ScriptEvaluatorConfig()
        elif not isinstance(evaluator_config, ScriptEvaluatorConfig):
            evaluator_config = ScriptEvaluatorConfig.model_validate(evaluator_config)

        try:
            self._set_logger()
            with ParslEvaluator(
                self._function,
                polling=evaluator_config.polling,
                max_submit=evaluator_config.max_submit,
                max_threads=evaluator_config.max_threads,
            ).with_htex(
                provider=provider,
                htex_kwargs=evaluator_config.htex_kwargs,
                worker_restart=evaluator_config.worker_restart,
            ).with_monitor(
                self._monitor,
            ) as evaluator:
                context = OptimizerContext(evaluator=evaluator, seed=self._seed)
                config = PlanConfig.model_validate(self._plan_config)
                plan = Plan(config, context)
                for event_type, function in (
                    (EventType.FINISHED_OPTIMIZER_STEP, self._log_exit_code),
                    *self._observers,
                ):
                    plan.add_observer(event_type, function)
                plan.run()
        finally:
            os.chdir(cwd)

        return plan
