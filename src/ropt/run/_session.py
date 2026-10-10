"""The session: the owner of the pools built on it.

A **session** is what a pool belongs to. Its factories are the only way to build
a pool with workers, and closing the session releases every pool it built, so
most code needs no further cleanup. It is also what a run's stop requests reach:
each run registers a signal while it lasts, and `stop()` sets them all.

A run is started on the session or on one of its pools, so what it belongs to is
stated at the call site and never discovered from the surroundings. Any number
of pools — and any number of sessions — can be open at once.
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
from ropt.enums import ExitCode
from ropt.exceptions import WorkflowError

from ._evaluate import _evaluate, _evaluate_batch
from ._optimize import _optimize, _optimize_many
from ._pool import WorkerPool

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import ArrayLike

    from ropt.components.concurrency import AbortSignal
    from ropt.components.event_handlers import EventHandler
    from ropt.components.executors import Executor
    from ropt.results import FunctionResults, GradientResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback
    from ._result import EvaluationResult, OptimizationResult

_CLOSED = (
    "This session is not open; build pools and start runs inside its `with` "
    "block, for example `with session() as s: pool = s.thread_pool()`."
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

    A run started on the session itself evaluates in-process, on the calling
    thread; a run started on one of its pools evaluates on that pool's workers.
    Either way the run belongs to this session.

    A session may be opened inside another, and pools from different sessions
    never interact. A session is single use — once closed it cannot be reopened.

    [`abort`][ropt.Session.abort] aborts the runs that belong to it,
    which is what a caller on another thread — a signal handler, a user
    interface — calls to bring them down.
    """

    def __init__(self) -> None:
        """Initialize the session."""
        # A driver thread may build its own pool while the session is closing.
        self._lock = threading.Lock()
        self._pools: list[WorkerPool] | None = None
        self._entered = False
        self._signals: set[AbortSignal] = set()

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
        """Close the session, aborting its runs and releasing every pool."""
        # Closing and taking the snapshot are one step: a run that registers
        # between them would be neither refused as late nor reached by the
        # abort, and would outlive the session.
        with self._lock:
            pools, self._pools = self._pools, None
            signals = list(self._signals)
        for signal in signals:
            signal.abort()
        for pool in pools or ():
            pool._release()  # ruff: ignore[private-member-access]

    def abort(self) -> None:
        """Abort the runs that belong to this session.

        Each run ends with `ExitCode.USER_ABORT`, keeping whatever its completed
        batches produced; one aborted during its first batch has no result. An
        [`offload`][ropt.WorkerPool.offload] in flight raises
        [`AbortedError`][ropt.exceptions.AbortedError] instead, since it has no
        result object to report a reason on.

        This reaches every run under way. Calling it is thread-safe, and safe on
        a session with no runs. See
        [Aborting a run from outside](../running/running.md#stopping-from-outside)
        for which runs are reached and when.
        """
        self._abort_all(ExitCode.USER_ABORT)

    def _fail(self) -> None:
        # A run that failed brings down the rest, which is what makes a script
        # stop at the first problem.
        self._abort_all(ExitCode.ABORTED_ON_ERROR)

    def _abort_all(self, exit_code: ExitCode) -> None:
        with self._lock:
            signals = list(self._signals)
        for signal in signals:
            signal.abort(exit_code)

    def _register(self, signal: AbortSignal) -> None:
        with self._lock:
            if self._pools is None:
                raise WorkflowError(_CLOSED)
            self._signals.add(signal)

    def _deregister(self, signal: AbortSignal) -> None:
        with self._lock:
            self._signals.discard(signal)

    def optimize(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any],
        x0: ArrayLike,
        function: EvaluationFunction,
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | None = None,
        constraint_tolerance: float = 1e-10,
        metadata: dict[str, Any] | None = None,
        f0: FunctionResults | None = None,
        g0: GradientResults | None = None,
        report_gradients: bool = False,
    ) -> OptimizationResult:
        """Run a single optimization in-process, on this session.

        The evaluations run on the calling thread. Start the run on one of this
        session's pools to give them workers. See
        [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            config:               The optimization configuration.
            x0:                   The initial variable vector.
            function:             The per-realization evaluation function.
            handlers:             Optional handlers, called in the order listed.
            report:               Optional callback invoked per evaluation.
            constraint_tolerance: The tolerance within which a constraint holds.
            metadata:             Optional dictionary attached to every result.
            f0:                   Optional function results at `x0`.
            g0:                   Optional gradient results at `x0`.
            report_gradients:     Whether `report` also receives gradient results.

        Returns:
            An [`OptimizationResult`][ropt.OptimizationResult].

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _optimize(
            self,
            None,
            config,
            x0,
            function,
            handlers=handlers,
            report=report,
            constraint_tolerance=constraint_tolerance,
            bundle_size=None,
            metadata=metadata,
            optimize_many_signal=None,
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )

    def optimize_many(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any] | Sequence[dict[str, Any]],
        x0: ArrayLike,
        function: EvaluationFunction | Sequence[EvaluationFunction],
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | Sequence[ReportCallback] | None = None,
        limit: int | None = None,
        constraint_tolerance: float = 1e-10,
        metadata: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        f0: FunctionResults | Sequence[FunctionResults | None] | None = None,
        g0: GradientResults | Sequence[GradientResults | None] | None = None,
        report_gradients: bool = False,
    ) -> tuple[OptimizationResult, ...]:
        """Run several optimizations concurrently in-process, on this session.

        The runs overlap on driver threads, but each evaluates on its own
        thread, so `function` is called by several threads at once and must
        tolerate that. Start them on one of this session's pools to give the
        evaluations workers instead. See
        [Evaluating in Parallel](../running/parallel.md) for a
        walkthrough.

        Args:
            config:               The configuration, or one per run.
            x0:                   The initial vector, or one per row.
            function:             The evaluation function, or one per run.
            handlers:             Optional handlers, fed by every run.
            report:               Optional callback, shared or one per run.
            limit:                The maximum number of runs at once.
            constraint_tolerance: The tolerance within which a constraint holds.
            metadata:             Optional dictionary attached to every result.
            f0:                   Optional function results at `x0`, shared or
                                  one per run.
            g0:                   Optional gradient results at `x0`, shared or
                                  one per run.
            report_gradients:     Whether `report` also receives gradient results.

        Returns:
            One [`OptimizationResult`][ropt.OptimizationResult] per run.

        Raises:
            RunsFailedError: If any of the runs raised.
            ValueError:      If `x0` has the wrong shape, or the sequences
                             given per run disagree in length.
            WorkflowError:   If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _optimize_many(
            self,
            None,
            config,
            x0,
            function,
            handlers=handlers,
            report=report,
            limit=limit,
            constraint_tolerance=constraint_tolerance,
            bundle_size=None,
            metadata=metadata,
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )

    def evaluate(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any],
        variables: ArrayLike,
        function: EvaluationFunction,
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult[FunctionResults | None]:
        """Evaluate a single variable vector in-process, without optimizing.

        See [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            config:    The optimization configuration.
            variables: The variable vector to evaluate.
            function:  The per-realization evaluation function.
            handlers:  Optional handlers, called in the order listed.
            report:    Optional callback invoked with the results.
            metadata:  Optional dictionary attached to the results.

        Returns:
            An [`EvaluationResult`][ropt.EvaluationResult] whose
            `results` is the [`FunctionResults`][ropt.results.FunctionResults]
            for the vector, or `None` if the evaluation was aborted.

        Raises:
            ValueError:    If `variables` is not a single vector.
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _evaluate(
            self,
            None,
            config,
            variables,
            function,
            handlers=handlers,
            report=report,
            bundle_size=None,
            metadata=metadata,
        )

    def evaluate_batch(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any],
        variables: ArrayLike,
        function: EvaluationFunction,
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult[tuple[FunctionResults, ...]]:
        """Evaluate a batch of variable vectors in-process, without optimizing.

        Each row of `variables` is one vector, and the results come back in the
        same order. See [Running Optimizations](../running/running.md) for a
        walkthrough.

        Args:
            config:    The optimization configuration.
            variables: The variable vectors to evaluate, one per row.
            function:  The per-realization evaluation function.
            handlers:  Optional handlers, called in the order listed.
            report:    Optional callback invoked with each evaluation.
            metadata:  Optional dictionary attached to every result.

        Returns:
            An [`EvaluationResult`][ropt.EvaluationResult] whose
            `results` holds one
            [`FunctionResults`][ropt.results.FunctionResults] per vector, and is
            empty if the batch was aborted.

        Raises:
            ValueError:    If `variables` is not a 2-D matrix.
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _evaluate_batch(
            self,
            None,
            config,
            variables,
            function,
            handlers=handlers,
            report=report,
            bundle_size=None,
            metadata=metadata,
        )

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

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
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
        being raised; use [`local_pool`][ropt.Session.local_pool] where
        an evaluation launches external programs.

        Args:
            workers:             The number of worker processes.
            max_tasks_per_child: Evaluations before a worker is replaced.
            bundle_size:         Evaluations per worker task, `0` for a batch.

        Returns:
            A pool backed by worker processes.

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
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
        [`hpc_pool`][ropt.Session.hpc_pool]: the same job shape, so an
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

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
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
        [`local_pool`][ropt.Session.local_pool] first: it has the same
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

        Raises:
            WorkflowError: If this session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
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


def session() -> Session:
    """Open a session that owns the pools built on it.

    Build pools with the session's factories, and start on them the runs that
    should evaluate there. Closing the session releases every pool it built, so
    most code needs no further cleanup:

    ```python
    with session() as s:
        pool = s.thread_pool(workers=4)
        result = pool.optimize(config, x0, objective)
    ```

    A run started with the module-level [`optimize`][ropt.optimize]
    evaluates in-process and needs no session. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    A run that fails stops the other runs on the session, which is what makes a
    script stop at the first problem. Its exception is raised where the run was
    started. A run or offload started from inside another run only raises; see
    [When an inner run fails](../running/nested.md#when-an-inner-run-fails).

    Returns:
        A context manager binding the [`Session`][ropt.Session].
    """
    return Session()
