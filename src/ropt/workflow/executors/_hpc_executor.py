"""Defines a class for running evaluations on a HPC cluster."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sysconfig
import tempfile
from collections import ChainMap
from importlib.util import find_spec
from pathlib import Path
from pickle import UnpicklingError  # noqa: S403
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from ropt._logging import get_logger
from ropt.exceptions import ExecutorFailure

from .base import ExecutorBase, Task

if TYPE_CHECKING:
    from uuid import UUID

_logger = get_logger(__name__)


_HAVE_HPC: Final = (
    find_spec("cloudpickle") is not None and find_spec("pysqa") is not None
)

if _HAVE_HPC:
    import cloudpickle
    import pysqa


class HPCExecutor(ExecutorBase):
    """An executor for submitting tasks to an HPC cluster.

    Interfaces with an HPC queueing system (e.g. Slurm) via `pysqa`.
    Requires `ropt[hpc]` to be installed.

    See [Parallel Evaluation](../usage/parallel.md#hpcexecutor) for full
    details on configuration and lifecycle.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        workdir: Path | str = "./",
        workers: int = 1,
        queue_size: int = 0,
        interval: float = 1,
        queue_type: str = "slurm",
        template: str | None = None,
        config_path: Path | str | None = None,
        cluster: str | None = None,
        queue: str | None = None,
        cores: int = 1,
        retries: int = 30,
        cleanup: bool = True,
    ) -> None:
        """Initialize the HPC executor.

        See [Parallel Evaluation](../usage/parallel.md#hpcexecutor) for
        configuration details.

        Args:
            workdir:     Shared filesystem directory for temporary I/O files.
            workers:     Maximum concurrent HPC jobs.
            queue_size:  Maximum task queue size (0 = unlimited).
            interval:    Polling interval in seconds.
            queue_type:  Queueing system type (e.g. `"slurm"`).
            template:    Optional submission script template string.
            config_path: Optional path to `pysqa` configuration directory.
            cluster:     Optional cluster name.
            queue:       Optional queue/partition name.
            cores:       CPUs per task.
            retries:     Number of polling retries before declaring a task failed.
            cleanup:     Whether to remove task files after result retrieval.

        Raises:
            RuntimeError: If neither a `template` is provided nor a valid
                          `config_path` can be found.
        """
        super().__init__(queue_size=queue_size)
        self._workdir = Path(workdir)
        if not self._workdir.is_absolute():
            msg = f"HPC working directory is not absolute: {self._workdir}"
            raise RuntimeError(msg)
        if not self._workdir.exists():
            msg = f"HPC work directory not found: {self._workdir}"
            raise RuntimeError(msg)
        self._workers = workers
        self._interval = interval
        self._queue = queue
        self._cores = cores
        self._retries_limit = retries
        self._cleanup = cleanup
        self._worker_task: asyncio.Task[None] | None = None

        self._template = template
        config_path = _get_config_path(config_path)
        if self._template is None and config_path is None:
            msg = "The HPC cluster has not been configured"
            raise RuntimeError(msg)
        if self._template is not None:
            self._queue_adapter = pysqa.QueueAdapter(queue_type=queue_type)
        else:
            assert config_path is not None
            self._queue_adapter = pysqa.QueueAdapter(
                directory=str(config_path / queue_type)
            )
        if cluster is not None:
            self._queue_adapter.switch_cluster(cluster)

        self._tasks: dict[str | UUID, Task] = {}
        self._results: dict[str | UUID, Any] = {}
        self._jobs: dict[str | UUID, int] = {}
        self._retries: dict[str | UUID, int] = {}

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.
        """
        self._worker_task = task_group.create_task(self._worker())
        _logger.info(
            "Starting HPC executor (%d max workers, %.1fs poll interval)",
            self._workers,
            self._interval,
        )
        await self._finish_start(task_group)

    async def _worker(self) -> None:
        while self._running.is_set():
            tasks: list[Task] = []
            try:
                for _ in range(max(self._workers - len(self._tasks), 0)):
                    tasks.append(self._task_queue.get_nowait())
                    self._task_queue.task_done()
            except asyncio.QueueEmpty:
                pass
            await asyncio.to_thread(self._run_worker_tasks, tasks)
            await asyncio.sleep(self._interval)

    def cleanup(self) -> None:
        """Clean up the executor resources."""
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        self._worker_task = None
        for task in self._tasks.values():
            task.cancel_all()
        self._drain_and_kill()

    def _run_worker_tasks(self, tasks: list[Task]) -> None:
        for task in tasks:
            self._submit(task)
        self._poll()
        self._get_results()

    def _submit(self, task: Task) -> None:
        task_id = task.name or uuid4()
        if task_id in ChainMap(self._tasks, self._results, self._jobs, self._retries):
            msg = "Task ID already in use, unique names required"
            raise RuntimeError(msg)
        self._tasks[task_id] = task
        input_file = self._workdir / f"{task_id}.in"
        output_file = self._workdir / f"{task_id}.out"
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=self._workdir)
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as fp:
                cloudpickle.dump((task.function, task.args, task.kwargs), fp)
                fp.flush()
                os.fsync(fp.fileno())
            tmp_path.rename(input_file)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        self._jobs[task_id] = self._queue_adapter.submit_job(
            job_name=task_id,
            output=f"{task_id}.txt",
            working_directory=str(self._workdir),
            command=f"python -m ropt.workflow.executors {input_file} {output_file}",
            submission_template=self._template,
            queue=self._queue,
            cores=self._cores,
        )
        _logger.debug("Submitted HPC job %s (job id: %s)", task_id, self._jobs[task_id])

    def _poll(self) -> None:
        try:
            jobs = self._queue_adapter.get_status_of_my_jobs()["jobid"].values
        except Exception:  # noqa: BLE001
            return
        for task_id in list(self._tasks):
            if self._jobs.get(task_id) is None or self._jobs[task_id] in jobs:
                continue
            output_file = self._workdir / f"{task_id}.out"
            try:
                with output_file.open("rb") as fp:
                    self._results[task_id] = cloudpickle.load(fp)
                self._retries.pop(task_id, None)
                del self._jobs[task_id]
            except FileNotFoundError:
                self._retries[task_id] = self._retries.get(task_id, 0) + 1
                if self._retries[task_id] >= self._retries_limit:
                    self._retries.pop(task_id, None)
                    del self._jobs[task_id]
                    msg = f"Output file for task {task_id} never appeared"
                    _logger.warning(
                        "HPC task %s failed: output file never appeared", task_id
                    )
                    self._results[task_id] = ExecutorFailure(msg)
            except (OSError, EOFError, UnpicklingError):
                retry_count = self._retries.get(task_id, 0) + 1
                self._retries[task_id] = retry_count
                if retry_count >= self._retries_limit:
                    self._retries.pop(task_id, None)
                    del self._jobs[task_id]
                    msg = f"No valid result for task {task_id} after {self._retries_limit} retries"
                    _logger.warning(
                        "HPC task %s failed: no valid result after %d retries",
                        task_id,
                        self._retries_limit,
                    )
                    self._results[task_id] = ExecutorFailure(msg)

    def _get_results(self) -> None:
        remove = []
        for task_id, task in self._tasks.items():
            if task_id in self._results:
                result = self._results.pop(task_id)
                if isinstance(result, Exception) and not isinstance(
                    result, ExecutorFailure
                ):
                    task.cancel_all()
                    raise result
                task.put_result(result)
                remove.append(task_id)
        for task_id in remove:
            self._tasks.pop(task_id)
            if self._cleanup:
                self._cleanup_files(task_id)

    def _cleanup_files(self, task_id: str | UUID) -> None:
        for suffix in (".in", ".out", ".txt"):
            path = self._workdir / f"{task_id}{suffix}"
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)


def _get_config_path(config_path: Path | str | None) -> Path | None:
    if config_path is None:
        path = Path(sysconfig.get_paths()["data"]) / "share" / "ropt" / "pysqa"
        if path.exists():
            return path
    else:
        return Path(config_path).resolve()
    return None
