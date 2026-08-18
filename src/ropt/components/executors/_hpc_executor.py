"""Defines a class for running evaluations on a HPC cluster."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sysconfig
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from pickle import UnpicklingError  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from ropt._logging import get_logger
from ropt.exceptions import ExecutionError, ExecutorFailure, WorkflowError

from .base import ExecutorBase, Submission, WorkItem

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

    See [Parallel Evaluation](../workflows/parallel.md#hpcexecutor) for full
    details on configuration and lifecycle.
    """

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        workdir: Path | str = "./",
        workers: int = 1,
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

        See [Parallel Evaluation](../workflows/parallel.md#hpcexecutor) for
        configuration details.

        Args:
            workdir:     Shared-filesystem directory for each work item's
                         serialized I/O files; also passed as the job working
                         directory (template-dependent). Work item files are
                         never overwritten, so concurrent executors need
                         distinct workdirs.
            workers:     Maximum concurrent HPC jobs.
            interval:    Polling interval in seconds.
            queue_type:  Queueing system type (e.g. `"slurm"`).
            template:    Optional submission script template string.
            config_path: Optional path to `pysqa` configuration directory.
            cluster:     Optional cluster name.
            queue:       Optional queue/partition name.
            cores:       CPUs per work item.
            retries:     Number of extra polls to wait for a work item's result
                         after the first attempt fails (`0` gives up at once).
            cleanup:     Whether to remove work item files once their result is
                         retrieved or their job is cancelled.

        For multi-cluster `pysqa` configurations the cluster is resolved as
        follows: if `cluster` is given it is selected directly, and if `queue`
        is also given it is verified to be available on that cluster. If only
        `queue` is given the cluster that provides it is derived automatically;
        this requires exactly one cluster to provide the queue.

        Raises:
            ExecutionError: If neither a `template` is provided nor a valid
                          `config_path` can be found, if the requested cluster
                          is unknown, if the queue is not available on the
                          requested cluster, if the queue cannot be resolved to
                          exactly one cluster, or if `workers`, `interval` or
                          `retries` is out of range.
        """
        super().__init__()
        self._workdir = Path(workdir)
        if not self._workdir.is_absolute():
            msg = f"The HPC working directory must be an absolute path: {self._workdir}"
            raise ExecutionError(msg)
        if not self._workdir.exists():
            msg = f"The HPC working directory does not exist: {self._workdir}"
            raise ExecutionError(msg)
        if workers < 1:
            msg = f"The number of HPC workers must be at least one: {workers}"
            raise ExecutionError(msg)
        if interval < 0:
            msg = f"The HPC polling interval must not be negative: {interval}"
            raise ExecutionError(msg)
        if retries < 0:
            msg = f"The number of HPC retries must not be negative: {retries}"
            raise ExecutionError(msg)
        self._workers = workers
        self._interval = interval
        self._queue = queue
        self._cores = cores
        self._retries_limit = retries
        self._remove_files = cleanup
        self._worker_task: asyncio.Task[None] | None = None
        self._pool: ThreadPoolExecutor | None = None

        self._template = template
        config_path = _get_config_path(config_path)
        if self._template is None and config_path is None:
            msg = "The HPC cluster is not configured; provide a template or a valid config_path."
            raise ExecutionError(msg)
        if self._template is not None:
            self._queue_adapter = pysqa.QueueAdapter(queue_type=queue_type)
        else:
            assert config_path is not None
            self._queue_adapter = pysqa.QueueAdapter(
                directory=str(config_path / queue_type)
            )
            _select_cluster(self._queue_adapter, cluster, queue)

        self._items: dict[str | UUID, tuple[Submission, WorkItem]] = {}
        self._jobs: dict[str | UUID, int] = {}
        self._retries: dict[str | UUID, int] = {}
        self._poll_failures = 0
        self._jobs_lock = threading.Lock()
        self._jobs_closed = False
        self._work_arrived = asyncio.Event()

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.
        """
        self._begin_start()
        self._work_arrived = asyncio.Event()
        with self._jobs_lock:
            self._jobs_closed = False
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ropt-hpc-poll"
        )
        self._worker_task = task_group.create_task(self._worker(self._pool))
        _logger.info(
            "Starting HPC executor (%d max workers, %.1fs poll interval)",
            self._workers,
            self._interval,
        )
        await self._finish_start(task_group)

    async def _worker(self, pool: ThreadPoolExecutor) -> None:
        loop = asyncio.get_running_loop()
        while self._running.is_set():
            pending = self._take_work_items()
            if self._items:
                results = await loop.run_in_executor(
                    pool, self._run_work_items, pending
                )
                self._deliver_results(results)
            await self._wait_for_work()

    async def _wait_for_work(self) -> None:
        self._work_arrived.clear()
        if len(self._items) < self._workers and not self._work_queue.empty():
            return
        idle = not self._items and self._work_queue.empty()
        timeout = None if idle else self._interval
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._work_arrived.wait(), timeout)

    def _accept(self, submission: Submission) -> None:
        super()._accept(submission)
        self._work_arrived.set()

    def _take_work_items(self) -> list[tuple[str | UUID, WorkItem]]:
        pending: list[tuple[str | UUID, WorkItem]] = []
        while len(self._items) < self._workers:
            try:
                submission, work_item = self._work_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if submission.is_finished:
                # Its caller has already left, so this would be a cluster job
                # whose result nobody reads.
                continue
            try:
                item_id = self._register(submission, work_item)
            except WorkflowError as exc:
                self._fail(submission, exc)
                continue
            pending.append((item_id, work_item))
        return pending

    def _register(self, submission: Submission, work_item: WorkItem) -> str | UUID:
        item_id = work_item.name or uuid4()
        if item_id in self._items:
            msg = f"Work item ID '{item_id}' is already in use; names must be unique."
            raise WorkflowError(msg)
        self._items[item_id] = (submission, work_item)
        return item_id

    def _deliver_results(self, results: dict[str | UUID, Any]) -> None:
        for item_id, result in results.items():
            entry = self._items.pop(item_id, None)
            if entry is None:
                continue
            submission, work_item = entry
            if isinstance(result, Exception) and not isinstance(
                result, ExecutorFailure
            ):
                # Deliver to the evaluator; keep the executor alive (no raise).
                self._fail(submission, result)
            else:
                self._deliver(submission, work_item, result)

    def _cleanup(self) -> None:
        """Clean up the executor resources."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        self._worker_task = None
        self._cancel_jobs()
        self._items.clear()
        self._cleanup_submissions()

    def _cancel_jobs(self) -> None:
        with self._jobs_lock:
            self._jobs_closed = True
            jobs = dict(self._jobs)
            self._jobs.clear()
        for item_id, job_id in jobs.items():
            self._delete_job(item_id, job_id)

    def _delete_job(self, item_id: str | UUID, job_id: int) -> None:
        try:
            self._queue_adapter.delete_job(job_id)
        except Exception as exc:  # ruff: ignore[blind-except]
            _logger.warning(
                "Could not cancel HPC job %s (job id: %s): %s", item_id, job_id, exc
            )
        else:
            _logger.debug("Cancelled HPC job %s (job id: %s)", item_id, job_id)
        self._retries.pop(item_id, None)
        if self._remove_files:
            self._cleanup_files(item_id)

    def _run_work_items(
        self, pending: list[tuple[str | UUID, WorkItem]]
    ) -> dict[str | UUID, Any]:
        results: dict[str | UUID, Any] = {}
        for item_id, work_item in pending:
            try:
                if not self._submit(item_id, work_item):
                    return results
            except Exception as exc:  # ruff: ignore[blind-except]
                results[item_id] = exc
        return results | self._poll()

    def _submit(self, item_id: str | UUID, work_item: WorkItem) -> bool:
        with self._jobs_lock:
            if self._jobs_closed:
                return False
        existing = any(
            (self._workdir / f"{item_id}{suffix}").exists()
            for suffix in (".in", ".out", ".txt")
        )
        if existing:
            msg = (
                f"Work item files for '{item_id}' already exist in {self._workdir}; "
                "give each executor its own working directory."
            )
            raise ExecutionError(msg)
        input_file = self._workdir / f"{item_id}.in"
        output_file = self._workdir / f"{item_id}.out"
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=self._workdir)
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as fp:
                cloudpickle.dump(
                    (work_item.function, work_item.args, work_item.kwargs), fp
                )
                fp.flush()
                os.fsync(fp.fileno())
            tmp_path.rename(input_file)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        try:
            job_id = self._queue_adapter.submit_job(
                job_name=item_id,
                output=f"{item_id}.txt",
                working_directory=str(self._workdir),
                command=f"python -m ropt.components.executors {input_file} {output_file}",
                submission_template=self._template,
                queue=self._queue,
                cores=self._cores,
            )
        except BaseException:
            if self._remove_files:
                self._cleanup_files(item_id)
            raise
        with self._jobs_lock:
            stopped = self._jobs_closed
            if not stopped:
                self._jobs[item_id] = job_id
        if stopped:
            self._delete_job(item_id, job_id)
            return False
        _logger.debug("Submitted HPC job %s (job id: %s)", item_id, job_id)
        return True

    def _poll(self) -> dict[str | UUID, Any]:
        results: dict[str | UUID, Any] = {}
        try:
            jobs = set(self._queue_adapter.get_status_of_my_jobs()["jobid"].tolist())
        except Exception as exc:  # ruff: ignore[blind-except]
            return self._handle_query_failure(exc)
        self._poll_failures = 0
        with self._jobs_lock:
            submitted = dict(self._jobs)
        for item_id, job_id in submitted.items():
            if job_id in jobs:
                continue
            output_file = self._workdir / f"{item_id}.out"
            try:
                with output_file.open("rb") as fp:
                    results[item_id] = cloudpickle.load(fp)
                self._retries.pop(item_id, None)
                self._drop_job(item_id)
            except FileNotFoundError:
                self._retries[item_id] = self._retries.get(item_id, 0) + 1
                if self._retries[item_id] > self._retries_limit:
                    self._retries.pop(item_id, None)
                    self._drop_job(item_id)
                    msg = f"Output file for work item {item_id} never appeared"
                    _logger.warning(
                        "HPC work item %s failed: output file never appeared", item_id
                    )
                    results[item_id] = ExecutorFailure(msg)
            except (OSError, EOFError, UnpicklingError):
                retry_count = self._retries.get(item_id, 0) + 1
                self._retries[item_id] = retry_count
                if retry_count > self._retries_limit:
                    self._retries.pop(item_id, None)
                    self._drop_job(item_id)
                    msg = f"No valid result for work item {item_id} after {self._retries_limit} retries"
                    _logger.warning(
                        "HPC work item %s failed: no valid result after %d retries",
                        item_id,
                        self._retries_limit,
                    )
                    results[item_id] = ExecutorFailure(msg)
        if self._remove_files:
            for item_id in results:
                self._cleanup_files(item_id)
        return results

    def _handle_query_failure(self, exc: BaseException) -> dict[str | UUID, Any]:
        results: dict[str | UUID, Any] = {}
        self._poll_failures += 1
        _logger.warning(
            "Querying the HPC scheduler failed (%d/%d): %s",
            self._poll_failures,
            self._retries_limit + 1,
            exc,
        )
        if self._poll_failures > self._retries_limit:
            msg = (
                "The HPC scheduler could not be queried after "
                f"{self._retries_limit + 1} attempts: {exc}"
            )
            with self._jobs_lock:
                submitted = list(self._jobs)
            for item_id in submitted:
                self._drop_job(item_id)
                self._retries.pop(item_id, None)
                results[item_id] = ExecutorFailure(msg)
            self._poll_failures = 0
        return results

    def _drop_job(self, item_id: str | UUID) -> None:
        with self._jobs_lock:
            self._jobs.pop(item_id, None)

    def _cleanup_files(self, item_id: str | UUID) -> None:
        for suffix in (".in", ".out", ".txt"):
            path = self._workdir / f"{item_id}{suffix}"
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


def _select_cluster(
    queue_adapter: pysqa.QueueAdapter, cluster: str | None, queue: str | None
) -> None:
    clusters = queue_adapter.list_clusters()
    if cluster is not None and cluster not in clusters:
        msg = f"Unknown HPC cluster: {cluster}."
        raise ExecutionError(msg)
    candidates = [cluster] if cluster is not None else clusters

    if queue is None:
        if cluster is not None:
            queue_adapter.switch_cluster(cluster)
        return

    matches = [
        name for name in candidates if _cluster_has_queue(queue_adapter, name, queue)
    ]
    if not matches:
        target = (
            f"HPC cluster '{cluster}'" if cluster is not None else "any HPC cluster"
        )
        msg = f"Queue '{queue}' is not available on {target}."
        raise ExecutionError(msg)
    if len(matches) > 1:
        cluster_names = ", ".join(matches)
        msg = (
            f"Queue '{queue}' is available on multiple HPC clusters: {cluster_names}. "
            "Specify a cluster."
        )
        raise ExecutionError(msg)
    queue_adapter.switch_cluster(matches[0])


def _cluster_has_queue(
    queue_adapter: pysqa.QueueAdapter, cluster: str, queue: str
) -> bool:
    queue_adapter.switch_cluster(cluster)
    queue_list = queue_adapter.queue_list
    return queue_list is not None and queue in queue_list
