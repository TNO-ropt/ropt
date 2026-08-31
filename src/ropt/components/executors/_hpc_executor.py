"""Defines a class for running evaluations on a HPC cluster.

A cluster job is what [`JobExecutorBase`][] calls a job: this module supplies the
three answers it needs — how to submit one through `pysqa`, how to ask the
scheduler which are still queued or running, and how to cancel one — plus the
cluster-specific configuration that goes with them.
"""

from __future__ import annotations

import sysconfig
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ropt.exceptions import ExecutionError

from ._job_executor import JobExecutorBase

if TYPE_CHECKING:
    from uuid import UUID

# `cloudpickle` is not required: without it, jobs may only be given functions
# the standard library can send, which is what `_serialize` falls back to.
_HAVE_HPC: Final = find_spec("pysqa") is not None

_DEFAULT_SCHEDULER: Final = "slurm"

_RESERVED_SUBMIT_OPTIONS: Final = frozenset(
    {
        "job_name",
        "output",
        "working_directory",
        "command",
        "submission_template",
        "queue",
        "cores",
        "memory_max",
        "run_time_max",
    }
)

if _HAVE_HPC:
    import pysqa


class HPCExecutor(JobExecutorBase):
    """An executor for submitting tasks to an HPC cluster.

    Interfaces with an HPC queueing system (for example Slurm) via `pysqa`.
    Requires `ropt[hpc]` to be installed.

    See [Parallel Evaluation](../workflows/parallel.md#hpcexecutor) for full
    details on configuration and lifecycle.
    """

    _kind = "HPC"
    _backend = "HPC scheduler"

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        workdir: Path | str,
        workers: int = 1,
        interval: float = 1,
        config_path: Path | str | None = None,
        cluster: str | None = None,
        queue: str | None = None,
        template: str | None = None,
        scheduler: str | None = None,
        cores: int = 1,
        memory_max: int | str | None = None,
        run_time_max: int | None = None,
        submit_options: dict[str, Any] | None = None,
        retries: int = 30,
        query_retries: int = 30,
        cleanup: bool = True,
    ) -> None:
        """Initialize the HPC executor.

        There are two ways to say how jobs are submitted, and they are mutually
        exclusive. Either a `pysqa` configuration supplies the clusters, the
        queues and their submission scripts — the normal case on an installed
        cluster, where only `queue` need be given — or a `template` supplies the
        submission script directly, in which case nothing is configured and
        `scheduler` names the queueing system.

        See [Parallel Evaluation](../workflows/parallel.md#hpcexecutor) for
        configuration details.

        Args:
            workdir:       Shared-filesystem directory for each work item's
                           serialized I/O files; also passed as the job working
                           directory (template-dependent). Must be an existing
                           absolute path: there is no default, because only the
                           caller knows which directory the cluster shares. Work
                           item files are never overwritten, so concurrent
                           executors need distinct workdirs.
            workers:       Maximum concurrent HPC jobs.
            interval:      Polling interval in seconds.
            config_path:   The `pysqa` configuration directory, the one holding
                           `queue.yaml` or `clusters.yaml`. Defaults to the
                           site-wide configuration installed alongside ropt.
            cluster:       Optional cluster name, when the configuration defines
                           more than one. Defaults to the configured primary.
            queue:         Optional name of a queue defined in the
                           configuration. This is not necessarily the
                           scheduler's partition name: it selects a queue entry,
                           whose submission script names the partition. Defaults
                           to the configured primary.
            template:      A submission script template, submitted instead of
                           any configuration. Everything the scheduler needs,
                           the partition included, must be in it.
            scheduler:     The queueing system a `template` is written for, for
                           example `"slurm"` (the default). Only meaningful with
                           a `template`, since a configuration names its own.
                           `pysqa` calls this `queue_type`.
            cores:         CPUs per work item.
            memory_max:    Memory per work item. Rendered by the submission
                           script; with a configuration it is also clamped to
                           the queue's limit.
            run_time_max:  Run time per work item, typically in seconds. With a
                           configuration, the queue's own limit applies when
                           this is not given.
            submit_options: Extra variables for the submission script, for
                           whatever the script declares beyond the standard
                           names — an account, a reservation, a GPU request.
                           Entries that are `None` are dropped, so omitting a
                           key and passing `None` mean the same thing.
            retries:       Number of extra polls to wait for a work item's
                           result after the first attempt fails (`0` gives up at
                           once). This is about the shared filesystem, not about
                           the scheduler.
            query_retries: Number of extra attempts to query the scheduler after
                           one fails (`0` gives up at once). A run this long
                           fails every job that is out, because nothing can be
                           said about a job that cannot be asked after.
            cleanup:       Whether to remove work item files once their result is
                           retrieved or their job is cancelled. A work item that
                           failed keeps its captured output, which is the only
                           record of why.

        Raises:
            ValueError:     If `workdir` is not an existing absolute path, if
                            `workers`, `interval`, `retries` or `query_retries`
                            is out of range, if arguments from both submission
                            modes are combined, or if `submit_options` names
                            something the executor already passes.
            ExecutionError: If no configuration can be found, if the requested
                            cluster is unknown, if the queue is not available on
                            the requested cluster, or if the queue cannot be
                            resolved to exactly one cluster.
        """
        workdir = Path(workdir)
        if not workdir.is_absolute():
            msg = f"The HPC working directory must be an absolute path: {workdir}"
            raise ValueError(msg)
        if not workdir.exists():
            msg = f"The HPC working directory does not exist: {workdir}"
            raise ValueError(msg)
        if query_retries < 0:
            msg = (
                f"The number of HPC query retries must not be negative: {query_retries}"
            )
            raise ValueError(msg)
        super().__init__(
            workdir=workdir,
            workers=workers,
            interval=interval,
            retries=retries,
            cleanup=cleanup,
        )
        self._queue = queue
        self._cores = cores
        self._memory_max = memory_max
        self._run_time_max = run_time_max
        self._submit_options = _checked_submit_options(submit_options)
        self._query_retries = query_retries

        self._template = template
        if template is None:
            if scheduler is not None:
                msg = (
                    "A scheduler applies to a template only; an HPC "
                    "configuration names its own queueing system."
                )
                raise ValueError(msg)
            resolved = _get_config_path(config_path)
            if resolved is None:
                msg = (
                    "The HPC cluster is not configured; "
                    "provide a config_path or a template."
                )
                raise ExecutionError(msg)
            self._queue_adapter = pysqa.QueueAdapter(directory=str(resolved))
            _select_cluster(self._queue_adapter, cluster, queue)
        else:
            _reject_configuration_arguments(config_path, cluster, queue)
            self._queue_adapter = pysqa.QueueAdapter(
                queue_type=_DEFAULT_SCHEDULER if scheduler is None else scheduler
            )

    def _start_job(self, item_id: str | UUID, command: list[str]) -> int:
        return int(
            self._queue_adapter.submit_job(
                job_name=item_id,
                output=f"{item_id}.txt",
                working_directory=str(self._workdir),
                command=" ".join(command),
                submission_template=self._template,
                queue=self._queue,
                cores=self._cores,
                memory_max=self._memory_max,
                run_time_max=self._run_time_max,
                **self._submit_options,
            )
        )

    def _live_job_ids(self) -> set[int]:
        # The only place that knows the scheduler answers with a table: above
        # this line a queueing system is a source of job ids and nothing else,
        # so `pandas` stays `pysqa`'s dependency rather than becoming ropt's.
        return set(self._queue_adapter.get_status_of_my_jobs()["jobid"].tolist())

    def _cancel_job(self, job_id: int) -> None:
        self._queue_adapter.delete_job(job_id)


def _checked_submit_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    reserved = sorted(_RESERVED_SUBMIT_OPTIONS.intersection(options))
    if reserved:
        msg = (
            "These HPC submit options are set by the executor itself: "
            f"{', '.join(reserved)}."
        )
        raise ValueError(msg)
    return {name: value for name, value in options.items() if value is not None}


def _reject_configuration_arguments(
    config_path: Path | str | None, cluster: str | None, queue: str | None
) -> None:
    given = [
        name
        for name, value in (
            ("config_path", config_path),
            ("cluster", cluster),
            ("queue", queue),
        )
        if value is not None
    ]
    if given:
        msg = (
            f"An HPC template submits without a configuration, it cannot be "
            f"combined with: {', '.join(given)}."
        )
        raise ValueError(msg)


def _get_config_path(config_path: Path | str | None) -> Path | None:
    if config_path is None:
        # Falls back to a site-wide configuration installed alongside ropt, so
        # that users on a configured cluster need not point at it themselves.
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
