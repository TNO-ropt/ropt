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
import threading
from contextvars import ContextVar, Token
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from ropt.components._loop import schedule
from ropt.components.concurrency import run_concurrent
from ropt.components.evaluators import BatchIdCounter
from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from ropt.components.event_handlers import EventDispatcher
    from ropt.components.executors import Executor

_T = TypeVar("_T")

_STOPPED = "The block's background session is not running; open a new block."


_active_session: ContextVar[_Session | None] = ContextVar(
    "ropt_simple_session", default=None
)


class _Session:
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
        self._stopped = threading.Event()
        self._task_group: asyncio.TaskGroup | None = None
        self._shutdown: asyncio.Event | None = None
        self._executor: Executor | None = None
        self._batch_counter: BatchIdCounter | None = None
        self._failure: BaseException | None = None

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait()

    def stop(self) -> None:
        assert self._thread is not None
        if self._shutdown is not None:
            schedule(self._loop, self._shutdown.set)
        self._thread.join()
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure

    def _thread_main(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        main_task = self._loop.create_task(self._run())
        try:
            self._loop.run_until_complete(main_task)
        except BaseException as exc:  # ruff: ignore[blind-except]
            self._failure = exc
        finally:
            # Normally already set by `_run`; this also covers a session that
            # died before it could get that far.
            self._stopped.set()
            self._ready.set()
            try:
                self._shut_down(main_task)
            except BaseException as exc:  # ruff: ignore[blind-except]
                if self._failure is None:
                    self._failure = exc
            finally:
                self._loop.close()

    def _shut_down(self, main_task: asyncio.Task[None]) -> None:
        assert self._loop is not None
        try:
            if self._shutdown is not None and not self._shutdown.is_set():
                self._loop.run_until_complete(self._shutdown.wait())
        finally:
            unfinished = asyncio.all_tasks(self._loop)
            for task in unfinished:
                task.cancel()
            try:
                if unfinished:
                    self._loop.run_until_complete(
                        asyncio.gather(*unfinished, return_exceptions=True)
                    )
            finally:
                for task in {main_task, *unfinished}:
                    if task.done() and not task.cancelled():
                        task.exception()

    async def _run(self) -> None:
        self._shutdown = asyncio.Event()
        try:
            async with asyncio.TaskGroup() as task_group:
                self._task_group = task_group
                self._ready.set()
                await self._shutdown.wait()
                if self._executor is not None:
                    self._executor.cancel()
        finally:
            # Set here, on the loop thread while the loop is still serving, so a
            # caller cannot pass the check in `_start_on_loop` and then hand work
            # to a loop that will not run it.
            self._stopped.set()

    def _start_on_loop(self, coro: Coroutine[Any, Any, None]) -> None:
        assert self._loop is not None
        if self._stopped.is_set():
            coro.close()
            raise WorkflowError(_STOPPED)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        except RuntimeError:
            coro.close()
            # Lost the race with the shutdown: the check above passed, but by
            # the time the work reached the loop, it or its task group was gone.
            if self._stopped.is_set():
                raise WorkflowError(_STOPPED) from None
            raise

    def _require_task_group(self) -> asyncio.TaskGroup:
        if self._stopped.is_set() or self._task_group is None:
            raise WorkflowError(_STOPPED)
        return self._task_group

    def open_executor(self, make_executor: Callable[[], Executor]) -> None:
        if self._executor is not None:
            msg = (
                "Only one execution block (threads/processes/hpc) can be open at "
                "a time; they cannot be nested."
            )
            raise WorkflowError(msg)
        task_group = self._require_task_group()
        executor = make_executor()
        self._start_on_loop(executor.start(task_group))
        self._executor = executor  # Only set on success.
        self._batch_counter = BatchIdCounter()

    def close_executor(self) -> None:
        executor = self._executor
        self._executor = None
        self._batch_counter = None
        if executor is not None:
            schedule(self._loop, executor.cancel)

    def get_executor(self) -> Executor | None:
        return self._executor

    def get_batch_counter(self) -> BatchIdCounter | None:
        return self._batch_counter

    def open_dispatcher(self, dispatcher: EventDispatcher) -> None:
        self._start_on_loop(dispatcher.start(self._require_task_group()))

    def close_dispatcher(self, dispatcher: EventDispatcher) -> None:
        schedule(self._loop, dispatcher.cancel)

    def gather_shared(
        self, jobs: Sequence[Callable[[], _T]], limit: int | None
    ) -> list[_T]:
        """Run jobs concurrently with this session shared, and collect results.

        See the module-level `gather_shared`, which is the entry point for
        callers outside this package.

        Args:
            jobs:  The zero-argument callables to run, one result each.
            limit: The maximum number to run at once, or ``None`` for no limit.

        Returns:
            The job results, in the order of ``jobs``.
        """

        def _run(job: Callable[[], _T]) -> _T:
            token = _active_session.set(self)
            try:
                return job()
            finally:
                _active_session.reset(token)

        return run_concurrent([partial(_run, job) for job in jobs], limit)


def _acquire_session() -> tuple[_Session, Token[_Session | None] | None]:
    session = _active_session.get()
    if session is not None:
        return session, None
    session = _Session()
    session.start()
    return session, _active_session.set(session)


def _release_session(session: _Session, token: Token[_Session | None] | None) -> None:
    if token is not None:
        try:
            session.stop()
        finally:
            _active_session.reset(token)


def current_session() -> _Session | None:
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


def current_batch_counter() -> BatchIdCounter | None:
    """Return the open execution block's batch ID counter.

    Every run in the block draws its batch IDs from this one counter, so runs
    that share the block's executor never produce the same batch ID. Without a
    block there is nothing to share with, and each evaluator counts on its own.

    Returns:
        The block's counter, or `None` when no execution block is open.
    """
    session = _active_session.get()
    return None if session is None else session.get_batch_counter()


def gather_shared(
    jobs: Sequence[Callable[[], _T]], limit: int | None = None
) -> list[_T]:
    """Run jobs concurrently on driver threads, sharing the open block.

    Each job runs on its own thread with the block's session set active, so
    `current_executor`, [`offload`][ropt.simple.offload] and a nested
    [`optimize`][ropt.simple.optimize] inside a job resolve to this block's
    executor. This is what [`optimize_many`][ropt.simple.optimize_many] is built
    on; use it directly to launch runs of your own concurrently.

    An open `handlers` block is not propagated, because a bare thread does not
    inherit context variables: read `current_handlers` on the calling thread and
    pass the scope into each job. `limit` bounds how many run at once. The first
    job to raise propagates its error at once (fail-fast); pending jobs are then
    skipped and any already running are abandoned, since a Python thread cannot
    be stopped from the outside.

    Args:
        jobs:  The zero-argument callables to run, one result each.
        limit: The maximum number to run at once, or `None` for no limit.

    Returns:
        The job results, in the order of `jobs`.

    Raises:
        WorkflowError: If no execution block is open.
    """
    session = _active_session.get()
    if session is None:
        msg = (
            "gather_shared() requires an execution block, "
            "e.g. `with ropt.threads(...):`."
        )
        raise WorkflowError(msg)
    return session.gather_shared(jobs, limit)
