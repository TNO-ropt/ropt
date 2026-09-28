"""The session: the owner of the pools built on it.

A **session** is what a pool belongs to. Its factories are the only way to build
a pool with workers, and closing the session releases every pool it built, so
most code needs no further cleanup.

Everything a session hands out is passed to a run explicitly, never discovered
by it, so any number of pools — and any number of sessions — can be open at
once, and nothing here is ambient.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Self

from ropt.components.executors import (
    HPCExecutor,
    LocalJobExecutor,
    ProcessExecutor,
    ThreadExecutor,
)
from ropt.exceptions import WorkflowError

from ._pool import WorkerPool, _ExecutorPool

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ropt.components.executors import Executor

_CLOSED = (
    "This session is not open; build pools inside its `with` block, for "
    "example `with session() as s: pool = s.thread_pool()`."
)

_REOPENED = (
    "This session was already opened and cannot be entered again; open a "
    "separate session."
)


class Session:
    """An open session, and the factories that build on it.

    A pool needs an owner that releases its workers, which is why pools are
    built here rather than constructed on their own. Everything a session
    creates is returned to the caller and passed on explicitly; nothing is
    discovered from the surroundings.

    Bind the session with `as` and call its factories inside the block. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    A session may be opened inside another, and pools from different sessions
    never interact. A session is single use — once closed it cannot be reopened.
    """

    def __init__(self) -> None:
        """Initialize the session."""
        # A driver thread may build its own pool while the session is closing.
        self._lock = threading.Lock()
        self._pools: list[_ExecutorPool] | None = None
        self._entered = False

    def __enter__(self) -> Self:
        """Open the session.

        Returns:
            The session itself.

        Raises:
            WorkflowError: If the session was already opened.
        """
        with self._lock:
            if self._entered:
                raise WorkflowError(_REOPENED)
            self._entered = True
            self._pools = []
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the session, releasing every pool it built."""
        with self._lock:
            pools, self._pools = self._pools, None
        for pool in pools or ():
            pool._release()  # ruff: ignore[private-member-access]

    def thread_pool(self, *, workers: int = 1, bundle_size: int = 1) -> WorkerPool:
        """Create a pool that runs evaluations in worker threads.

        Use this where an evaluation spends its time waiting, or in a library
        that releases the GIL. See
        [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            workers:     The number of worker threads.
            bundle_size: Evaluations per worker task, `0` for a whole batch.

        Returns:
            A pool backed by worker threads.
        """
        return self._open_pool(
            lambda: ThreadExecutor(workers=workers, bundle_size=bundle_size)
        )

    def process_pool(
        self,
        *,
        workers: int = 1,
        max_tasks_per_child: int | None = None,
        bundle_size: int = 1,
    ) -> WorkerPool:
        """Create a pool that runs evaluations in worker processes.

        The evaluation function must be picklable. See
        [Running Optimizations](../running/running.md) for a walkthrough.

        Releasing the pool terminates its worker processes and nothing else. A
        program an evaluation started itself keeps running, without an error
        being raised; use [`local_pool`][ropt.simple.Session.local_pool] where
        an evaluation launches external programs.

        Args:
            workers:             The number of worker processes.
            max_tasks_per_child: Evaluations before a worker is replaced.
            bundle_size:         Evaluations per worker task, `0` for a batch.

        Returns:
            A pool backed by worker processes.
        """
        return self._open_pool(
            lambda: ProcessExecutor(
                workers=workers,
                max_tasks_per_child=max_tasks_per_child,
                bundle_size=bundle_size,
            )
        )

    def local_pool(  # ruff: ignore[too-many-arguments]
        self,
        *,
        workdir: Path | str | None = None,
        workers: int = 1,
        interval: float = 0.1,
        retries: int = 0,
        cleanup: bool = True,
        bundle_size: int = 1,
    ) -> WorkerPool:
        """Create a pool that runs each evaluation as a separate local process.

        Between a process pool and a cluster: every evaluation gets an
        interpreter of its own, which can be stopped outright and whose output
        is captured to a file, but there is no queueing system and nothing to
        install. This is the local stand-in for
        [`hpc_pool`][ropt.simple.Session.hpc_pool]: the same job shape, so an
        evaluation function that works here works there.

        Each job is a fresh command rather than a re-import of your script, so
        the evaluation function must live in a module the job can import, or
        the `ropt[cloudpickle]` extra must be installed.

        POSIX only. See [Running Optimizations](../running/running.md) for a
        walkthrough.

        Args:
            workdir:     The working directory, or `None` for a temporary one.
            workers:     The maximum number of concurrent jobs.
            interval:    Seconds between checks on the running jobs.
            retries:     Times a failed job is retried.
            cleanup:     Whether to remove the job files afterwards.
            bundle_size: Evaluations per job, `0` for a whole batch.

        Returns:
            A pool backed by local job processes.
        """
        return self._open_pool(
            lambda: LocalJobExecutor(
                workdir=workdir,
                workers=workers,
                interval=interval,
                retries=retries,
                cleanup=cleanup,
                bundle_size=bundle_size,
            )
        )

    def hpc_pool(  # ruff: ignore[too-many-arguments]
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
    ) -> WorkerPool:
        """Create a pool that runs evaluations on an HPC cluster.

        Interfaces with a cluster queue (for example Slurm) through `pysqa`;
        requires the `ropt[hpc]` extra. Each evaluation is a job started as its
        own command, so the evaluation function must live in a module the
        compute nodes can import, or the `ropt[cloudpickle]` extra must be
        installed. Develop against
        [`local_pool`][ropt.simple.Session.local_pool] first: it has the same
        shape and the same rule, without a cluster. The cluster is selected from
        `cluster`/`queue`: give a queue to search for its cluster, a cluster to
        use its default queue, or both to be explicit.

        A `template` is the alternative to all of that: it submits without a
        configuration, so it cannot be combined with `config_path`, `cluster` or
        `queue`. See [Running on an HPC
        cluster](../running/parallel.md#running-on-an-hpc-cluster) for the
        configuration layout and Slurm examples.

        Args:
            workdir:        The shared-filesystem working directory.
            workers:        The maximum number of concurrent cluster jobs.
            interval:       Seconds between polls of the cluster.
            config_path:    The `pysqa` configuration directory.
            cluster:        The cluster name.
            queue:          The queue name.
            template:       A submission-script template.
            scheduler:      The queueing system a `template` is written for.
            cores:          The number of CPUs per job.
            memory_max:     The memory per job.
            run_time_max:   The run time per job.
            submit_options: Extra variables for the submission script.
            retries:        Times a failed job is retried.
            query_retries:  Times the cluster is re-polled for a result.
            cleanup:        Whether to remove the job files afterwards.
            bundle_size:    Evaluations per job, `0` for a whole batch.

        Returns:
            A pool backed by an HPC cluster.
        """
        return self._open_pool(
            lambda: HPCExecutor(
                workdir=workdir,
                workers=workers,
                interval=interval,
                config_path=config_path,
                cluster=cluster,
                queue=queue,
                template=template,
                scheduler=scheduler,
                cores=cores,
                memory_max=memory_max,
                run_time_max=run_time_max,
                submit_options=submit_options,
                retries=retries,
                query_retries=query_retries,
                cleanup=cleanup,
                bundle_size=bundle_size,
            )
        )

    def _open_pool(self, make_executor: Callable[[], Executor]) -> WorkerPool:
        self._require_open()
        # Built outside the lock, because starting workers is slow. A session
        # that closes meanwhile is caught when the pool is registered.
        pool = _ExecutorPool(make_executor())
        with self._lock:
            if self._pools is None:
                pool._release()  # ruff: ignore[private-member-access]
                raise WorkflowError(_CLOSED)
            self._pools.append(pool)
        return pool

    def _require_open(self) -> None:
        with self._lock:
            if self._pools is None:
                raise WorkflowError(_CLOSED)


def session() -> Session:
    """Open a session that owns the pools built on it.

    Build pools with the session's factories, and pass them to the runs that
    should use them. Closing the session releases every pool it built, so most
    code needs no further cleanup:

    ```python
    with session() as s:
        pool = s.thread_pool(workers=4)
        result = optimize(config, x0, objective, pool=pool)
    ```

    A run given no pool evaluates in-process and needs no session. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Returns:
        A context manager binding the [`Session`][ropt.simple.Session].
    """
    return Session()
