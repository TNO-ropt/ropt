"""Defines a class for running evaluations on a HPC cluster."""

from __future__ import annotations

import sysconfig
from importlib.util import find_spec
from pathlib import Path
from pickle import UnpicklingError  # noqa: S403
from typing import TYPE_CHECKING, Final, TypeVar

from .base import Task

if TYPE_CHECKING:
    from uuid import UUID

    from .base import Task
import asyncio
from typing import TYPE_CHECKING

from .base import ServerBase, Task

if TYPE_CHECKING:
    import queue
    from uuid import UUID


R = TypeVar("R")
TR = TypeVar("TR")

_HAVE_HPC: Final = (
    find_spec("cloudpickle") is not None and find_spec("pysqa") is not None
)

if _HAVE_HPC:
    import cloudpickle
    import pysqa


class DefaultHPCServer(ServerBase[Task[R, TR]]):
    """A server for submitting tasks to a High-Performance Computing (HPC) cluster.

    This server interfaces with an HPC queueing system (like Slurm) via the `pysqa`
    library. It manages the entire lifecycle of a remote task, including:

    - Serializing the task (function and arguments) and writing it to a shared
      filesystem.
    - Submitting the task as a job to the HPC queue.
    - Polling the queue for the job's status.
    - Retrieving the results (or any exceptions) once the job is complete.

    Configuration of the cluster connection is handled either through a submission
    script template or a `pysqa` configuration directory.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        workdir: Path | str,
        workers: int = 1,
        maxsize: int = 0,
        interval: float = 1,
        queue_type: str = "slurm",
        template: str | None = None,
        config_path: Path | str | None = None,
        cluster: str | None = None,
    ) -> None:
        """Initialize the HPC server.

        This sets up the server for communication with an HPC cluster. The
        connection can be configured in two ways:

        1.  By providing a `template` string for the job submission script.
        2.  By providing a `config_path` to a directory containing `pysqa`
            cluster configurations.

        If `config_path` is not given, the server will look for a default
        configuration at `<sysconfig_data>/share/ropt/pysqa`. One of these
        configuration methods is required.

        Args:
            workdir:     Working directory on a shared filesystem accessible by
                         both the client and the HPC nodes. Used for temporary
                         input/output files.
            workers:     The maximum number of concurrent jobs to run on the HPC
                         cluster.
            maxsize:     The maximum number of tasks to hold in the internal
                         queue before submission. A value of 0 means an
                         unlimited size.
            interval:    The interval in seconds at which to poll the HPC queue
                         for job status updates.
            queue_type:  The type of the queueing system (e.g., "slurm"). This is
                         passed to `pysqa` and is also used to find the
                         correct subdirectory within `config_path`.
            template:    An optional submission script template. If provided, it
                         will be used by `pysqa` to generate the job submission
                         script.
            config_path: An optional path to a directory containing `pysqa`
                         cluster configuration files. This is used if `template`
                         is not provided.
            cluster:     Optional name of the cluster to use. If supported by the
                         installation, this makes it possible to switch between
                         clusters.

        Raises:
            RuntimeError: If neither a `template` is provided nor a valid
                          `config_path` can be found.
        """
        super().__init__(maxsize=maxsize)
        self._workdir = Path(workdir).resolve()
        self._workers = workers
        self._interval = interval
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

        self._tasks: dict[UUID, Task[R, TR]] = {}
        self._results: dict[UUID, R | None] = {}
        self._jobs: dict[UUID, int] = {}
        self._retries: dict[UUID, int] = {}

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the server.

        Args:
            task_group: The task group to use.
        """
        self._worker_task = task_group.create_task(self._worker())
        await self._finish_start(task_group)

    async def _worker(self) -> None:
        while self._running.is_set():
            tasks: list[Task[R, TR]] = []
            try:
                for _ in range(max(self._workers - len(self._tasks), 0)):
                    tasks.append(self._task_queue.get_nowait())
                    self._task_queue.task_done()
            except asyncio.QueueEmpty:
                pass
            await asyncio.to_thread(self._run_worker_tasks, tasks)
            await asyncio.sleep(self._interval)

    def cleanup(self) -> None:
        """Clean up the server resources.

        This method cancels the background worker task that polls the HPC queue
        and ensures that any clients waiting for results are notified of the
        shutdown.
        """
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        self._worker_task = None
        queues: set[queue.Queue[TR | None]] = set()
        for task in self._tasks.values():
            if task.result_queue not in queues:
                queues.add(task.result_queue)
                task.put_result(None)
        self._drain_and_kill()

    def _run_worker_tasks(self, tasks: list[Task[R, TR]]) -> None:
        for task in tasks:
            self._submit(task)
        self._poll()
        self._get_results()

    def _submit(self, task: Task[R, TR]) -> None:
        self._tasks[task.id] = task
        input_file = self._workdir / f"{task.id}.in"
        output_file = self._workdir / f"{task.id}.out"
        with input_file.open("wb") as fp:
            cloudpickle.dump((task.function, task.args, task.kwargs), fp)
        self._jobs[task.id] = self._queue_adapter.submit_job(
            job_name=task.id,
            output=f"{task.id}.txt",
            working_directory=str(self._workdir),
            command=f"python -m ropt.plugins.server {input_file} {output_file}",
            submission_template=self._template,
        )

    def _poll(self, retries: int = 2) -> None:
        jobs = self._queue_adapter.get_status_of_my_jobs()["jobid"].values
        for task_id in self._tasks:
            if self._jobs[task_id] in jobs:
                continue
            output_file = self._workdir / f"{task_id}.out"
            if output_file.exists():
                try:
                    if output_file.stat().st_size > 0:
                        with output_file.open("rb") as fp:
                            self._results[task_id] = cloudpickle.load(fp)
                        self._retries.pop(task_id, None)
                        del self._jobs[task_id]
                except (EOFError, UnpicklingError) as exc:
                    if self._retries.get(task_id, 0) >= retries:
                        self._results[task_id] = None
                        self._retries.pop(task_id, None)
                        del self._jobs[task_id]
                        msg = f"No result found for task {task_id}"
                        raise RuntimeError(msg) from exc
                    self._retries[task_id] = self._retries.get(task_id, 0) + 1

    def _get_results(self) -> None:
        remove = []
        for task_id, task in self._tasks.items():
            if task_id in self._results:
                result = self._results.pop(task_id)
                if isinstance(result, Exception):
                    task.put_result(None)
                    raise result
                task.put_result(result)
                remove.append(task_id)
        for task_id in remove:
            self._tasks.pop(task_id)


def _get_config_path(config_path: Path | str | None) -> Path | None:
    if config_path is None:
        path = Path(sysconfig.get_paths()["data"]) / "share" / "ropt" / "pysqa"
        if path.exists():
            return path
    else:
        return Path(config_path).resolve()
    return None
