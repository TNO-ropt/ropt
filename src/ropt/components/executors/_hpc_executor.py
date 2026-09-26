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

    See [Parallel Evaluation](../advanced/parallel.md#hpcexecutor) for full
    details on configuration and lifecycle.
    """

    _kind = "HPC"
    _backend_name = "HPC scheduler"

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
        bundle_size: int = 1,
    ) -> None:
        """Initialize the HPC executor.

        There are two ways to say how jobs are submitted, and they are mutually
        exclusive. Either a `pysqa` configuration supplies the clusters, the
        queues and their submission scripts — the normal case on an installed
        cluster, where only `queue` need be given — or a `template` supplies the
        submission script directly, in which case nothing is configured and
        `scheduler` names the queueing system.

        `workdir` must be an existing absolute path, and concurrently running
        executors need distinct ones: work item files are never overwritten. A
        `queue` is not a scheduler partition; it selects a configured entry
        whose submission script names the partition. `retries` covers a result
        the shared filesystem does not yet show, `query_retries` a scheduler
        that cannot be reached; the grace period each allows is its count times
        `interval`.

        `config_path` is the directory holding `queue.yaml` or `clusters.yaml`,
        and defaults to the configuration installed alongside `ropt`; `cluster`
        and `queue` default to the configured primary. `scheduler` is what
        `pysqa` calls `queue_type`, defaults to `"slurm"`, and is meaningful
        only with a `template`, which must itself carry everything the scheduler
        needs, the partition included.

        With a configuration, `cores` is clamped to the queue's minimum and
        maximum, `run_time_max` to its maximum and defaults to it when not
        given, and a numeric `memory_max` to its maximum — a string such as
        `"4G"` is passed through unchanged. Entries of `submit_options` that are
        `None` are dropped, so omitting a key and passing `None` mean the same
        thing; a name the executor sets itself is rejected. A work item that
        failed keeps its captured output even under `cleanup`, which is the only
        record of why.

        See [Parallel Evaluation](../advanced/parallel.md#hpcexecutor) for
        configuration details.

        Args:
            workdir:        Shared-filesystem directory for each work item's files.
            workers:        Maximum concurrent HPC jobs.
            interval:       Polling interval in seconds.
            config_path:    The `pysqa` configuration directory.
            cluster:        Optional cluster name, for a multi-cluster configuration.
            queue:          Optional name of a queue defined in the configuration.
            template:       A submission script template, used instead of a configuration.
            scheduler:      The queueing system a `template` is written for.
            cores:          CPUs per work item.
            memory_max:     Memory per work item.
            run_time_max:   Run time per work item, typically in seconds.
            submit_options: Extra variables for the submission script.
            retries:        Extra polls to wait for a work item's result.
            query_retries:  Extra attempts to query the scheduler after one fails.
            cleanup:        Whether to remove a work item's files once it settles.
            bundle_size:    Calls per job, `0` for a whole batch.

        Raises:
            ValueError:     If an argument is out of range or the modes are mixed.
            ExecutionError: If the configuration, cluster or queue cannot be resolved.
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
        # Set before the base builds its shared state, which reads it.
        self._query_retries = query_retries
        super().__init__(
            workdir=workdir,
            workers=workers,
            interval=interval,
            retries=retries,
            cleanup=cleanup,
            bundle_size=bundle_size,
        )
        self._queue = queue
        self._cores = cores
        self._memory_max = memory_max
        self._run_time_max = run_time_max
        self._submit_options = _checked_submit_options(submit_options)

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

    def _start_job(self, item_id: UUID, command: list[str]) -> int:
        return int(
            self._queue_adapter.submit_job(
                job_name=str(item_id),
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
