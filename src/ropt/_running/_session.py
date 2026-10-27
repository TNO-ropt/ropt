"""The session: the owner of the pools it builds.

A **session** is what a pool belongs to. Its factories are the only way to build
a pool with workers, and closing the session releases every pool it built, so
most code needs no further cleanup. It is also what an abort reaches: the
session has an abort scope of its own, which is the current scope in its
`with` block and in which every call given one of its pools is nested, and
`abort()` aborts it.

A call without a pool is nested in the current scope where it is made, which in
the `with` block is the scope of the session. Any number of pools — and any
number of sessions — can be open at once.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from ropt._concurrency import AbortScope, _grace, current_abort_scope
from ropt._executors import (
    HPCExecutor,
    LocalJobExecutor,
    ProcessExecutor,
    ThreadExecutor,
)
from ropt._executors._hpc_executor import clusters_with_queue
from ropt.enums import ExitCode
from ropt.exceptions import ExecutionError, WorkflowError

from ._pool import WorkerPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt._executors import Executor


_LOCAL_QUEUE = "local"

_CLOSED = (
    "This session is not open; build pools and start work inside its `with` "
    "block, for example `with session() as s: pool = s.thread_pool()`."
)

_REOPENED = (
    "This session was already opened and cannot be entered again; open a "
    "separate session."
)


class Session:
    """An open session, and the factories that build its pools.

    A pool needs an owner that releases its workers, which is why pools are
    built here rather than constructed on their own. Everything a session
    creates is returned to the caller and passed on explicitly; nothing is
    discovered from the surroundings.

    Bind the session with `as` and call its factories inside the block. See
    [Sessions](../running/sessions.md) for an overview.

    An optimization, evaluation or `run` call given a pool of the session
    runs on that pool's workers and belongs to this session. One without a pool
    in the session's `with` block, or inside a call made there, runs in-process
    and is aborted with the session.

    A session may be opened inside another, and pools from different sessions
    never interact; a session opened inside another session's `with` block, or
    inside a call made there, is aborted with it. A session is single use —
    once closed it cannot be reopened.

    [`abort`][ropt.Session.abort] aborts the calls given its pools or made in
    its `with` block, which is what a caller on another thread — a signal handler,
    a user interface — calls to bring them down.
    """

    def __init__(self) -> None:
        """Initialize the session."""
        # A driver thread may build its own pool while the session is closing.
        self._lock = threading.Lock()
        self._pools: list[WorkerPool] | None = None
        self._entered = False
        self._abort_scope = AbortScope()
        self._exit_stack = contextlib.ExitStack()

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
        self._exit_stack.enter_context(
            self._abort_scope.nested_in(current_abort_scope())
        )
        self._exit_stack.enter_context(self._abort_scope.as_current())
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the session, aborting its calls and releasing every pool.

        Work still running with its pools is waited for, five seconds at most
        in total, which gives the programs it started with
        [`run_program`][ropt.run_program] the time to end. Worker processes
        still busy after that are killed; threads are no longer waited for.
        """
        self._exit_stack.close()
        with self._lock:
            pools, self._pools = self._pools, None
        # The scope stays aborted, so a call that passed the check for an open
        # session just before the close is aborted as soon as it starts.
        self._abort_scope.abort()
        deadline = time.monotonic() + _grace.GRACE_PERIOD
        # Every pool is stopped before any is waited for, so that they all
        # share the one grace period.
        executors = [pool._release() for pool in pools or ()]  # ruff: ignore[private-member-access]
        for executor in executors:
            if executor is not None:
                executor._finish_shutdown(deadline)  # ruff: ignore[private-member-access]

    def abort(self) -> None:
        """Abort the calls given its pools or made in its `with` block.

        Each optimization, evaluation and `run` call under way raises
        [`AbortedError`][ropt.exceptions.AbortedError] with
        `ExitCode.USER_ABORT`. The session stays aborted: a call given one of
        its pools afterwards, or made in its `with` block, is aborted as soon as it starts.

        Calling it is thread-safe, and safe when nothing is running.
        See [Aborting from outside](../running/multiple_runs.md#stopping-from-outside)
        for which calls are reached and when.
        """
        self._abort_scope.abort(ExitCode.USER_ABORT)
        with self._lock:
            pools = list(self._pools or ())
        for pool in pools:
            pool._retire_after_abort()  # ruff: ignore[private-member-access]

    def _open_abort_scope(self) -> AbortScope:
        self._require_open()
        return self._abort_scope

    def thread_pool(self, *, workers: int = 1) -> WorkerPool:
        """Create a pool that runs evaluations in worker threads.

        Use this where an evaluation spends its time waiting, or in a library
        that releases the GIL. See
        [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            workers: The number of worker threads.

        Returns:
            A pool backed by worker threads.

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return self._open_pool(lambda: ThreadExecutor(workers=workers))

    def process_pool(
        self,
        *,
        workers: int = 1,
        max_tasks_per_child: int | None = None,
    ) -> WorkerPool:
        """Create a pool that runs evaluations in worker processes.

        The evaluation function must be picklable. See
        [Running Optimizations](../running/running.md) for a walkthrough.

        An abort reaches an evaluation that is still running, as it does on a
        thread pool: [`aborting`][ropt.aborting] turns `True`, and a program
        started with [`run_program`][ropt.run_program] is terminated. Workers
        still busy five seconds after an abort, or after the session closes,
        are killed.

        Args:
            workers:             The number of worker processes.
            max_tasks_per_child: Evaluations before a worker is replaced.

        Returns:
            A pool backed by worker processes.

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return self._open_pool(
            lambda: ProcessExecutor(
                workers=workers,
                max_tasks_per_child=max_tasks_per_child,
            )
        )

    def hpc_pool(  # ruff: ignore[too-many-arguments]
        self,
        *,
        workdir: Path | str,
        workers: int = 1,
        interval: float | None = None,
        config_path: Path | str | None = None,
        cluster: str | None = None,
        queue: str | None = None,
        template: str | None = None,
        scheduler: str | None = None,
        cores: int = 1,
        memory_max: int | str | None = None,
        run_time_max: int | None = None,
        submit_options: dict[str, Any] | None = None,
        extra_polls: int | None = None,
        query_retries: int = 30,
        cleanup: bool = True,
    ) -> WorkerPool:
        """Create a pool that runs evaluations on an HPC cluster.

        Interfaces with a cluster queue (for example Slurm) through `pysqa`;
        requires the `ropt[hpc]` extra. Each evaluation is a job started as its
        own command, so the evaluation function must live in a module the
        compute nodes can import, or the `ropt[cloudpickle]` extra must be
        installed. The cluster is selected from `cluster`/`queue`: give a queue
        to search for its cluster, a cluster to use its default queue, or both
        to be explicit.

        The queue `"local"` without a `cluster` runs each job as a process on
        this machine instead, with the same job files and the same rule for the
        evaluation function. It needs neither `pysqa` nor a configuration, and
        it ignores `cores`, `memory_max`, `run_time_max`, `submit_options` and
        `query_retries`. It is an error if a configured cluster also has a
        queue named `"local"`; pass that `cluster` to submit to it.

        A `template` is the alternative to a configuration: it cannot be
        combined with `config_path`, `cluster` or `queue`. See [Running on an
        HPC cluster](../running/parallel.md#running-on-an-hpc-cluster) for the
        configuration layout and Slurm examples.

        Args:
            workdir:        The shared-filesystem working directory.
            workers:        The maximum number of concurrent cluster jobs.
            interval:       Seconds between polls of the cluster (default:
                            1, or 0.1 for local jobs).
            config_path:    The `pysqa` configuration directory.
            cluster:        The cluster name.
            queue:          The queue name.
            template:       A submission-script template.
            scheduler:      The queueing system a `template` is written for.
            cores:          The number of CPUs per job.
            memory_max:     The memory per job.
            run_time_max:   The run time per job.
            submit_options: Extra variables for the submission script.
            extra_polls:    Extra polls to wait for a work item's result
                            (default: 30, or 0 for local jobs).
            query_retries:  Times the cluster is re-polled for a result.
            cleanup:        Whether to remove the job files afterwards.

        Returns:
            A pool backed by an HPC cluster, or by local job processes.

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        if queue == _LOCAL_QUEUE and cluster is None:
            return self._open_pool(
                lambda: _local_queue_executor(
                    workdir=workdir,
                    workers=workers,
                    interval=0.1 if interval is None else interval,
                    extra_polls=0 if extra_polls is None else extra_polls,
                    cleanup=cleanup,
                    config_path=config_path,
                    template=template,
                    scheduler=scheduler,
                )
            )
        return self._open_pool(
            lambda: HPCExecutor(
                workdir=workdir,
                workers=workers,
                interval=1 if interval is None else interval,
                config_path=config_path,
                cluster=cluster,
                queue=queue,
                template=template,
                scheduler=scheduler,
                cores=cores,
                memory_max=memory_max,
                run_time_max=run_time_max,
                submit_options=submit_options,
                extra_polls=30 if extra_polls is None else extra_polls,
                query_retries=query_retries,
                cleanup=cleanup,
            )
        )

    def _open_pool(self, make_executor: Callable[[], Executor]) -> WorkerPool:
        self._require_open()
        # Built outside the lock, because starting workers is slow. A session
        # that closes meanwhile is caught when the pool is registered.
        pool = WorkerPool(self, make_executor())
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


def _local_queue_executor(  # ruff: ignore[too-many-arguments]
    *,
    workdir: Path | str,
    workers: int,
    interval: float,
    extra_polls: int,
    cleanup: bool,
    config_path: Path | str | None,
    template: str | None,
    scheduler: str | None,
) -> LocalJobExecutor:
    if template is not None or scheduler is not None:
        msg = "The local queue cannot be combined with a template or a scheduler."
        raise ValueError(msg)
    if not Path(workdir).is_absolute():
        msg = f"The HPC working directory must be an absolute path: {workdir}"
        raise ValueError(msg)
    clusters = clusters_with_queue(config_path, _LOCAL_QUEUE)
    if clusters:
        msg = (
            f"Queue '{_LOCAL_QUEUE}' is ambiguous: it is defined on HPC "
            f"cluster(s) {', '.join(clusters)}. Specify a cluster to submit to it."
        )
        raise ExecutionError(msg)
    return LocalJobExecutor(
        workdir=workdir,
        workers=workers,
        interval=interval,
        extra_polls=extra_polls,
        cleanup=cleanup,
    )


def session() -> Session:
    """Open a session that owns the pools it builds.

    Build pools with the session's factories, and pass a pool to the
    optimizations and evaluations that should use its workers. Closing the
    session releases every pool it built, so most code needs no further cleanup:

    ```python
    with session() as s:
        pool = s.thread_pool(workers=4)
        result = optimize(config, x0, objective, pool=pool)
    ```

    An optimization started with [`optimize`][ropt.optimize] without a pool
    evaluates in-process and needs no session. See
    [Sessions](../running/sessions.md) for an overview.

    A call that fails aborts the work it started, and its exception is raised
    where the call was made. Not caught, it ends the `with` block, and closing
    the session aborts whatever is still running.
    [Failures and Aborts](../running/failures.md) describes each case.

    Returns:
        A context manager binding the [`Session`][ropt.Session].
    """
    return Session()
