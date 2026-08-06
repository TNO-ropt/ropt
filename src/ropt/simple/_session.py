"""The background session: a loop thread, a task group, an executor.

A session opens one background event loop (on its own daemon thread) with a
long-lived `TaskGroup`. The evaluation executor is started on that task group
when the session opens and runs for the session's lifetime. A user-code error is
re-raised out of the compute step (on the calling thread) and leaves the
executor running, so the session stays usable and the executor is reused.

The active session is held in a single contextvar slot; sessions do not nest.
With no session open, the high-level entry points run sequentially on the
calling thread.
"""

from __future__ import annotations

import asyncio
import threading
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Self, TypeVar, cast

from ropt.components.evaluators import FunctionEvaluator, ParallelEvaluator
from ropt.components.executors import (
    Executor,
    MultiprocessingExecutor,
    ThreadingExecutor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike

    from ropt.components.compute_steps import ComputeStep
    from ropt.components.evaluators import EvaluationFunctionCallback, Evaluator
    from ropt.context import EnOptContext
    from ropt.enums import ExitCode

_T = TypeVar("_T")


_active_session: ContextVar[Session | None] = ContextVar(
    "ropt_simple_session", default=None
)


class Session:
    """A background loop, task group, and a single worker pool."""

    def __init__(self, make_executor: Callable[[], Executor]) -> None:
        self._make_executor = make_executor
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._token: Token[Session | None] | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._shutdown: asyncio.Event | None = None
        self._executor: Executor | None = None
        self._start_error: BaseException | None = None

    def __enter__(self) -> Self:
        if _active_session.get() is not None:
            msg = "Sessions do not nest; a session is already active in this context."
            raise RuntimeError(msg)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._start_error is not None:
            self._thread.join()
            raise self._start_error
        self._token = _active_session.set(self)
        return self

    def __exit__(self, *_exc: object) -> None:
        assert self._token is not None
        _active_session.reset(self._token)
        assert self._shutdown is not None
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._shutdown.set)
        assert self._thread is not None
        self._thread.join()

    def _thread_main(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._execute_task_group())
        finally:
            self._loop.close()

    async def _execute_task_group(self) -> None:
        self._shutdown = asyncio.Event()
        async with asyncio.TaskGroup() as task_group:
            self._task_group = task_group
            try:
                executor = self._make_executor()
                await executor.start(task_group)
            except Exception as exc:  # ruff: ignore[blind-except]
                # Surface a start failure to __enter__ instead of deadlocking it.
                self._start_error = exc
                self._ready.set()
                return
            self._executor = executor
            self._ready.set()
            await self._shutdown.wait()
            if executor.is_running():
                executor.cancel()

    def get_executor(self) -> Executor:
        assert self._executor is not None
        return self._executor

    def gather(self, jobs: Sequence[Callable[[], _T]], limit: int | None) -> list[_T]:
        assert self._loop is not None
        try:
            return asyncio.run_coroutine_threadsafe(
                self._gather_coro(jobs, limit), self._loop
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
        jobs: Sequence[Callable[[], _T]], limit: int | None
    ) -> list[_T]:
        semaphore = asyncio.Semaphore(max(limit if limit is not None else len(jobs), 1))
        results: list[_T | None] = [None] * len(jobs)

        async def _run_job(index: int, job: Callable[[], _T]) -> None:
            async with semaphore:
                results[index] = await asyncio.to_thread(job)

        # Fail-fast: the first job to raise cancels the others' awaits and its
        # error propagates out at once; the orphaned to_thread jobs run on.
        async with asyncio.TaskGroup() as task_group:
            for index, job in enumerate(jobs):
                task_group.create_task(_run_job(index, job))
        return cast("list[_T]", results)


def threads(*, workers: int = 1) -> Session:
    """Run evaluations in a thread pool for the duration of the block.

    See [High-Level API](../usage/simple.md) for a walkthrough.

    Args:
        workers: The number of worker threads.

    Returns:
        A session context manager backing evaluations with threads.
    """
    return Session(lambda: ThreadingExecutor(workers=workers))


def processes(*, workers: int = 1) -> Session:
    """Run evaluations in a process pool for the duration of the block.

    The objective must be picklable. See
    [High-Level API](../usage/simple.md) for a walkthrough.

    Args:
        workers: The number of worker processes.

    Returns:
        A session context manager backing evaluations with processes.
    """
    return Session(lambda: MultiprocessingExecutor(workers=workers))


def current_session() -> Session | None:
    """Return the session active in the current context, if any.

    Returns:
        The active session, or `None` when running on the sequential floor.
    """
    return _active_session.get()


def make_evaluator(
    callback: EvaluationFunctionCallback, session: Session | None
) -> Evaluator:
    """Build the evaluator for the given session.

    Args:
        callback: The adapted per-evaluation callback.
        session:  The session whose pool to use, or `None` for the sequential
                  floor.

    Returns:
        A `ParallelEvaluator` bound to the session's pool, or a
        `FunctionEvaluator` when there is no session.
    """
    if session is None:
        return FunctionEvaluator(function=callback)
    return ParallelEvaluator(function=callback, executor=session.get_executor())


def run_step(
    step: ComputeStep,
    *,
    context: EnOptContext,
    variables: ArrayLike,
) -> ExitCode:
    """Run a compute step on the calling thread.

    The step's evaluator is already wired to the session's executor (if any) at
    construction time, so running the step needs no session.

    Args:
        step:      The compute step to run.
        context:   The optimizer context.
        variables: The initial variable vector(s).

    Returns:
        The exit code returned by the step.
    """
    return cast("ExitCode", step.run(context=context, variables=variables))
