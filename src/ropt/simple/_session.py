"""The background session: a loop thread, a task group, and an optional executor.

Any high-level block (an execution manager such as `threads()`/`processes()`)
opens a background event loop on its own daemon thread, with a long-lived
`TaskGroup`. The **session** is that loop and task group; it is established by
the first block to open and torn down by that same owner when it exits, so
nested blocks reuse one loop.

The **executor** is a separate layer: an execution manager installs one on the
session's task group and removes it on exit. At most one executor is active at a
time — opening a second execution manager while one is active raises (nested
executors are not supported). The evaluator for each `optimize()`/`evaluate()`
call is chosen from whatever executor is active at that moment, or a sequential
`FunctionEvaluator` when none is.

The active session is held in a single contextvar slot. With no block open, the
high-level entry points run sequentially on the calling thread.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from contextvars import ContextVar, Token
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ropt.components.executors import (
    Executor,
    HPCExecutor,
    MultiprocessingExecutor,
    ThreadingExecutor,
)
from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike

    from ropt.components.compute_steps import ComputeStep
    from ropt.components.evaluators import EvaluationFunctionContext, NameCallback
    from ropt.components.event_handlers import EventDispatcher
    from ropt.context import EnOptContext
    from ropt.enums import ExitCode

_T = TypeVar("_T")


_active_session: ContextVar[Session | None] = ContextVar(
    "ropt_simple_session", default=None
)


class Session:
    """A background event loop and task group hosting at most one executor.

    The session (loop + task group) is opened by the first high-level block and
    reused by nested blocks; the owning block tears it down on exit. An execution
    manager installs an executor via `open_executor` and removes it via
    `close_executor` — at most one may be active at a time.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._task_group: asyncio.TaskGroup | None = None
        self._shutdown: asyncio.Event | None = None
        self._executor: Executor | None = None
        self._run_counter = itertools.count()

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait()

    def stop(self) -> None:
        assert self._loop is not None
        assert self._shutdown is not None
        assert self._thread is not None
        self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join()

    def _thread_main(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()

    async def _run(self) -> None:
        self._shutdown = asyncio.Event()
        async with asyncio.TaskGroup() as task_group:
            self._task_group = task_group
            self._ready.set()
            await self._shutdown.wait()
            # Defensive: a scope normally removes its executor before the session
            # stops; cancel a leftover so the task group can exit.
            if self._executor is not None and self._executor.is_running():
                self._executor.cancel()

    def open_executor(self, make_executor: Callable[[], Executor]) -> None:
        if self._executor is not None:
            msg = (
                "Only one executor may be active at a time; nested execution "
                "managers are not supported."
            )
            raise WorkflowError(msg)
        assert self._loop is not None
        assert self._task_group is not None
        executor = make_executor()
        # Start on the loop thread and block until ready, surfacing a start
        # failure (e.g. a process pool that cannot spawn) on the caller's stack.
        asyncio.run_coroutine_threadsafe(
            executor.start(self._task_group), self._loop
        ).result()
        self._executor = executor
        # Restart run numbering so task names are unique within this executor.
        self._run_counter = itertools.count()

    def next_run_id(self) -> int:
        return next(self._run_counter)

    def close_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            assert self._loop is not None
            self._loop.call_soon_threadsafe(executor.cancel)

    def get_executor(self) -> Executor | None:
        return self._executor

    def open_dispatcher(self, dispatcher: EventDispatcher) -> None:
        assert self._loop is not None
        assert self._task_group is not None
        asyncio.run_coroutine_threadsafe(
            dispatcher.start(self._task_group), self._loop
        ).result()

    def close_dispatcher(self, dispatcher: EventDispatcher) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(dispatcher.cancel)

    def gather(self, jobs: Sequence[Callable[[], _T]], limit: int | None) -> list[_T]:
        """Run jobs concurrently on the session's loop and return their results.

        Each job runs on its own worker thread; ``limit`` bounds how many run at
        once. The first job to raise cancels the others and its error propagates
        (fail-fast).

        Args:
            jobs:  The zero-argument callables to run, one result each.
            limit: The maximum number to run at once, or ``None`` for no limit.

        Returns:
            The job results, in the order of ``jobs``.
        """
        assert self._loop is not None
        try:
            return asyncio.run_coroutine_threadsafe(
                self._gather_coro(self, jobs, limit), self._loop
            ).result()
        except BaseExceptionGroup as exc:
            # TaskGroup wraps failures in a group; unwrap to the first leaf so
            # the caller sees the job's original error, like the sequential path.
            leaf: BaseException = exc
            while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
                leaf = leaf.exceptions[0]
            raise leaf from None

    @staticmethod
    async def _gather_coro(
        session: Session, jobs: Sequence[Callable[[], _T]], limit: int | None
    ) -> list[_T]:
        semaphore = asyncio.Semaphore(max(limit if limit is not None else len(jobs), 1))
        results: list[_T | None] = [None] * len(jobs)

        async def _run_job(index: int, job: Callable[[], _T]) -> None:
            async with semaphore:
                results[index] = await asyncio.to_thread(
                    _run_with_active_session, session, job
                )

        # Fail-fast: the first job to raise cancels the others' awaits and its
        # error propagates out at once; the orphaned to_thread jobs run on.
        async with asyncio.TaskGroup() as task_group:
            for index, job in enumerate(jobs):
                task_group.create_task(_run_job(index, job))
        return cast("list[_T]", results)


def _run_with_active_session(session: Session, job: Callable[[], _T]) -> _T:
    # gather() runs each job on its own worker thread, which does not inherit the
    # _active_session contextvar. Set it so current_executor()/offload() used
    # inside the job resolve the session, as on the thread that opened the block.
    token = _active_session.set(session)
    try:
        return job()
    finally:
        _active_session.reset(token)


def _acquire_session() -> tuple[Session, Token[Session | None] | None]:
    session = _active_session.get()
    if session is not None:
        return session, None
    session = Session()
    session.start()
    return session, _active_session.set(session)


def _release_session(session: Session, token: Token[Session | None] | None) -> None:
    if token is not None:
        session.stop()
        _active_session.reset(token)


class _ExecutionScope:
    """Install an executor on the ambient session for the duration of a block."""

    def __init__(self, make_executor: Callable[[], Executor]) -> None:
        self._make_executor = make_executor
        self._session: Session | None = None
        self._token: Token[Session | None] | None = None

    def __enter__(self) -> None:
        session, token = _acquire_session()
        try:
            session.open_executor(self._make_executor)
        except BaseException:
            # Roll back a session this scope created before re-raising.
            _release_session(session, token)
            raise
        self._session = session
        self._token = token

    def __exit__(self, *_exc: object) -> None:
        assert self._session is not None
        self._session.close_executor()
        _release_session(self._session, self._token)


def threads(*, workers: int = 1) -> _ExecutionScope:
    """Run evaluations in a thread pool for the duration of the block.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        workers: The number of worker threads.

    Returns:
        A context manager backing evaluations with a thread pool.
    """
    return _ExecutionScope(lambda: ThreadingExecutor(workers=workers))


def processes(*, workers: int = 1) -> _ExecutionScope:
    """Run evaluations in a process pool for the duration of the block.

    The objective must be picklable. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        workers: The number of worker processes.

    Returns:
        A context manager backing evaluations with a process pool.
    """
    return _ExecutionScope(lambda: MultiprocessingExecutor(workers=workers))


def hpc(  # ruff: ignore[too-many-arguments]
    *,
    workers: int = 1,
    cores: int = 1,
    cluster: str | None = None,
    queue: str | None = None,
    workdir: Path | str | None = None,
    config_path: Path | str | None = None,
    template: str | None = None,
    queue_type: str = "slurm",
) -> _ExecutionScope:
    """Run evaluations on an HPC cluster for the duration of the block.

    Interfaces with a cluster queue (e.g. Slurm) through `pysqa`; requires the
    `ropt[hpc]` extra, and the objective must be picklable. The cluster is
    selected from `cluster`/`queue`: give a queue to search for its cluster, a
    cluster to use its default queue, or both to be explicit. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        workers:     The maximum number of concurrent cluster jobs.
        cores:       The number of CPUs per job.
        cluster:     The cluster name, when the `pysqa` config defines several.
        queue:       The queue or partition name.
        workdir:     The shared-filesystem working directory (defaults to the
                     current directory).
        config_path: The path to the `pysqa` configuration directory.
        template:    An inline submission-script template, instead of a config.
        queue_type:  The queueing system type.

    Returns:
        A context manager backing evaluations with an HPC cluster.
    """
    resolved = Path.cwd() if workdir is None else Path(workdir).resolve()
    return _ExecutionScope(
        lambda: HPCExecutor(
            workers=workers,
            cores=cores,
            cluster=cluster,
            queue=queue,
            workdir=resolved,
            config_path=config_path,
            template=template,
            queue_type=queue_type,
        )
    )


def current_session() -> Session | None:
    """Return the session active in the current context, if any.

    Returns:
        The active session, or `None` when running on the sequential floor.
    """
    return _active_session.get()


def current_executor() -> Executor | None:
    """Return the active executor pool, or `None` on the sequential floor.

    Call this from a compute step to build its evaluator: with a pool, wrap it in
    a `ParallelEvaluator`; a `None` result means evaluate directly in-process
    with a `FunctionEvaluator`. Read it on the thread that owns the block and
    pass the result to each run, so `optimize_many`'s worker threads use the same
    pool.

    Returns:
        The active executor pool, or `None` when no execution block is open.
    """
    session = _active_session.get()
    return None if session is None else session.get_executor()


def run_step(
    step: ComputeStep,
    *,
    context: EnOptContext,
    variables: ArrayLike,
    metadata: dict[str, Any] | None = None,
) -> ExitCode:
    """Run a compute step on the calling thread.

    The step's evaluator is already wired to the session's executor (if any) at
    construction time, so running the step needs no session.

    Args:
        step:      The compute step to run.
        context:   The optimizer context.
        variables: The initial variable vector(s).
        metadata:  Optional dictionary attached to every emitted
                   [`Results`][ropt.results.Results].

    Returns:
        The exit code returned by the step.
    """
    return cast(
        "ExitCode",
        step.run(context=context, variables=variables, metadata=metadata),
    )


def _name_task(run_id: int, contexts: Sequence[EvaluationFunctionContext]) -> str:
    context = contexts[0]
    name = f"run{run_id}-b{context.batch_id}-r{context.realization}"
    if context.perturbation >= 0:
        name = f"{name}-p{context.perturbation}"
    return name


def make_task_namer(
    session: Session | None, executor: Executor | None
) -> NameCallback | None:
    """Build an auto-naming callback for a single run's tasks.

    Names have the form `run{id}-b{batch}-r{realization}[-p{perturbation}]`,
    where `id` is unique within the executor and the `-p` suffix is dropped for
    unperturbed evaluations. Only the `HPCExecutor` uses these names; for other
    executors the callback is harmless.

    Args:
        session:  The active session, or `None` on the sequential floor.
        executor: The active executor, or `None` on the sequential floor.

    Returns:
        A naming callback, or `None` when there is no executor.
    """
    if session is None or executor is None:
        return None
    return partial(_name_task, session.next_run_id())
