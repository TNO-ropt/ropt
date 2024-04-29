"""A class for running optimizations with script-based jobs."""

import json
import logging
import os
from collections import abc, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from string import Template
from traceback import format_exception
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
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

from ropt.config.enopt import EnOptConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.evaluator import EvaluatorContext
from ropt.evaluator.parsl import ParslEvaluator, State, Task
from ropt.events import OptimizationEvent
from ropt.exceptions import ConfigError
from ropt.optimization import EnsembleOptimizer
from ropt.results import FunctionResults


def _make_dict(
    variables: NDArray[np.float64], names: Sequence[Tuple[str, ...]]
) -> DefaultDict[str, Any]:
    def recursive_dict() -> DefaultDict[str, Any]:
        return defaultdict(recursive_dict)

    var_dict = recursive_dict()
    for idx, var_name in enumerate(names):
        tmp = var_dict
        for name in var_name[:-1]:
            tmp = tmp[str(name)]
        tmp[str(var_name[-1])] = variables[idx]

    return var_dict


def _write_dict(var_dict: Dict[str, Any], path: Path, name: str, depth: int) -> None:
    if depth == 0 or not isinstance(var_dict, abc.Mapping):
        filename = (path / name if name else path / "variables").with_suffix(".json")
        with filename.open("w", encoding="utf-8") as file_obj:
            json.dump(var_dict, file_obj, indent=2)
    else:
        for key, value in var_dict.items():
            _write_dict(value, path, f"{name}_{key}" if name else key, depth - 1)


def variables_to_json(
    variables: NDArray[np.float64], names: Tuple[Any, ...], path: Path, depth: int = 0
) -> None:
    """Export a vector of variables with given names to json.

    The `names` parameter must provide the name of each variable stored in
    variables. Each name should be a tuple, which will be used to generate a
    nested dictionary entry with the  variable value. The generated dict is
    stored in one or more json files depending on the value of `depth`. If
    `depth == 0`, the dictionary is saved as a single file with the name
    `variables.json`. If `depth > 0` files are generated by generating a json
    file for each entry given by the first `depth`th components of the name. The
    name of each file is constructed by concatenating the first `depth`th name
    components.

    The `level` parameter must be at least zero. The `path` parameters must
    denote an existing directory.

    Args:
        variables: The variables to export
        names:     The names of the variables
        path:      The directory to store the file
        depth:     The depth at which to generate files
    """
    msg = ""
    if depth < 0:
        msg = "The depth parameter must be at least zero"
    if not path.is_dir():
        msg = "The path parameters should point to a valid directory"
    if variables.ndim != 1:
        msg = "The variables parameter must be a 1D vector"
    if len(names) != variables.size:
        msg = "The length of the names parameter must match the variables size"
    if msg:
        raise ValueError(msg)
    return _write_dict(_make_dict(variables, names), path, "", depth)


@dataclass
class ScriptTask(Task):
    """Task class for use by the ScriptBasedOptimizer class."""

    name: str = ""
    job_labels: Tuple[str, ...] = field(default_factory=tuple)
    objective_paths: Tuple[Path, ...] = field(default_factory=tuple)
    constraint_paths: Tuple[Path, ...] = field(default_factory=tuple)
    logged: bool = False
    realization: int = 0
    _objectives: Optional[NDArray[np.float64]] = None
    _constraints: Optional[NDArray[np.float64]] = None

    def _read_file(self, path: Path) -> float:
        try:
            with path.open("r", encoding="utf-8") as file_obj:
                return float(file_obj.read())
        except BaseException as exc:  # noqa: BLE001
            self.exception = exc
            return np.nan

    def _read_results(self) -> None:
        if self.objective_paths:
            self._objectives = np.zeros(len(self.objective_paths), dtype=np.float64)
            for idx, path in enumerate(self.objective_paths):
                self._objectives[idx] = self._read_file(path)
        if self.constraint_paths:
            self._constraints = np.zeros(len(self.constraint_paths), dtype=np.float64)
            for idx, path in enumerate(self.constraint_paths):
                self._constraints[idx] = self._read_file(path)

    def get_objectives(self) -> Optional[NDArray[np.float64]]:
        """Get the objective values.

        Returns:
            The objective values.
        """
        self._read_results()
        return self._objectives

    def get_constraints(self) -> Optional[NDArray[np.float64]]:
        """Get the constraint values.

        Returns:
            The constraint values.
        """
        self._read_results()
        return self._constraints


@no_type_check
@bash_app()
def run_script(
    _: Any,  # noqa: ANN401
    script: str,
    stdout: TextIO,  # noqa: ARG001
    stderr: TextIO,  # noqa: ARG001
) -> str:
    return script


def get_function_files(config: EnOptConfig) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Default function to retrieve names of the files with evaluation results.

    Args:
        config: Optimizer configuration file.

    Returns:
        Two tuples with filenames of objective and constraint functions.
    """
    assert config.objective_functions.names is not None
    objective_names = tuple(str(name) for name in config.objective_functions.names)
    if config.nonlinear_constraints is not None:
        assert config.nonlinear_constraints.names is not None
        constraint_names = tuple(
            str(name) for name in config.nonlinear_constraints.names
        )
    else:
        constraint_names = ()
    return objective_names, constraint_names


class ScriptOptimizer:
    """Optimizer class for running script bases optimization workflows."""

    def __init__(  # noqa: PLR0913
        self,
        plan: Sequence[Dict[str, Any]],
        tasks: List[Tuple[str, str]],
        work_dir: Union[Path, str],
        *,
        job_dir: Optional[Union[Path, str]] = None,
        job_labels: Tuple[str, ...] = ("B{batch:04d}", "J{job:04d}"),
        variable_export_depth: int = 0,
        get_function_files: Callable[
            [EnOptConfig], Tuple[Tuple[str, ...], Tuple[str, ...]]
        ] = get_function_files,
        callbacks: Optional[Dict[EventType, Callable[..., None]]] = None,
        seed: Optional[int] = None,
        provider: Optional[ExecutionProvider] = None,
        max_threads: int = 4,
    ) -> None:
        """Initialize the optimizer.

        The directory where the jobs run is constructed from by nesting
        directories according the `job_labels` tuple. These directories are
        located in the directory where the optimizer runs, which can be set
        using the `work_dir` parameter of the `run()` method.

        Args:
            plan:                  The optimization plan to run
            tasks:                 A list of tuples mapping task names to
                                   strings containing bash code
            work_dir:              Working directory
            job_dir:               The directory to store files generated during
                                   optimization
            job_labels:            Label formats to use in generating filenames
                                   reports
            variable_export_depth: The depth parameter for exporting variables
            get_function_files:    Callable to retrieve function file names
            callbacks:             Dictionary of callbacks for the optimizer
            seed:                  Seed for the random number generator
            provider:              The provider that executes the jobs
            max_threads:           Maximum number of threads for local runs
        """
        self._plan = plan
        self._tasks = tasks
        self._work_dir = Path(work_dir).resolve()
        self._job_labels = job_labels
        self._variable_export_depth = variable_export_depth
        self._get_function_files = get_function_files
        self._callbacks = callbacks
        self._seed = seed
        self._provider = provider
        self._max_threads = max_threads
        self._status: Dict[int, Any] = {}
        self._optimal_result: Optional[FunctionResults] = None

        self._job_dir = self._work_dir if job_dir is None else Path(job_dir)
        if not self._job_dir.is_absolute():
            self._job_dir = self._work_dir / self._job_dir

    def _set_logger(self) -> None:
        self._logger = logging.getLogger("ScriptBasedOptimizer")
        self._logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("optimizer.log")
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def _workflow(
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
            for label in self._job_labels
        )

        path = self._job_dir / Path(*job_labels)
        Path.mkdir(path, parents=True, exist_ok=True)

        assert context.config.variables.names is not None
        variables_to_json(
            variables[job_idx],
            context.config.variables.names,
            path,
            depth=self._variable_export_depth,
        )

        tasks: List[ScriptTask] = []
        for task_name, script in self._tasks:
            substituted_script = Template(script).safe_substitute(
                work_dir=self._work_dir, realization=realization
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

        objective_names, constraint_names = self._get_function_files(context.config)
        tasks[-1].objective_paths = tuple(path / name for name in objective_names)
        tasks[-1].constraint_paths = tuple(path / name for name in constraint_names)

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
        with (self._work_dir / "status.txt").open("w", encoding="utf-8") as file_obj:
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
        with (self._work_dir / "states.json").open("w", encoding="utf-8") as file_obj:
            json.dump(states_json, file_obj, sort_keys=True, indent=4)

    def _monitor(self, batch_id: int, jobs: Dict[int, List[ScriptTask]]) -> None:
        self._write_state_report(batch_id, jobs)
        self._update_current_state(batch_id, jobs)
        self._store_states()

    def _log_exit_code(self, event: OptimizationEvent) -> None:
        exit_code = event.exit_code
        assert event.config is not None
        if exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS:
            self._logger.warning(
                "Too few successful realizations: optimization stopped.",
            )
        elif exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED:
            self._logger.warning(
                "Maximum number of functions reached: optimization stopped.",
            )
        elif exit_code == OptimizerExitCode.USER_ABORT:
            self._logger.warning("Optimization aborted by the user.")
        elif exit_code == OptimizerExitCode.OPTIMIZER_STEP_FINISHED:
            self._logger.info("Optimization finished normally.")

    def _handle_finished_optimizer(self, event: OptimizationEvent) -> None:
        self._log_exit_code(event)

    def run(self) -> Dict[str, Optional[FunctionResults]]:
        """Run the optimization."""
        cwd = Path.cwd()
        Path.mkdir(self._work_dir, parents=True, exist_ok=True)
        os.chdir(self._work_dir)

        try:
            self._set_logger()
            evaluator = ParslEvaluator(
                self._workflow,
                monitor=self._monitor,
                provider=self._provider,
                max_threads=self._max_threads,
            )
            optimizer = EnsembleOptimizer(evaluator)
            optimizer.add_observer(
                EventType.FINISHED_OPTIMIZER_STEP,
                self._handle_finished_optimizer,
            )
            if self._callbacks is not None:
                for event, callback in self._callbacks.items():
                    optimizer.add_observer(event, callback)
            optimizer.start_optimization(self._plan, seed=self._seed)
        finally:
            os.chdir(cwd)

        return optimizer.results


def _format_list(values: List[int]) -> str:
    grouped = (
        tuple(y for _, y in x)
        for _, x in groupby(enumerate(sorted(values)), lambda x: x[0] - x[1])
    )
    return ", ".join(
        "-".join([str(sub_group[0]), str(sub_group[-1])])
        if len(sub_group) > 1
        else str(sub_group[0])
        for sub_group in grouped
    )
