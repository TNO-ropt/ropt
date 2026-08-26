"""The background session: a loop thread, a task group, and the pools on it.

A **session** is a background event loop on its own daemon thread, with a
long-lived `TaskGroup`. The public [`session`][ropt.simple.session] block opens
one and hands out the pools built on it; closing the block releases the pools
and stops the loop. The same holds for the shared-handler groups built on it.

Everything a session hands out is passed to a run explicitly, never discovered
by it, so any number of pools and groups — and any number of sessions — can be
open at once, and nothing here is ambient.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

from ropt.components._loop import schedule
from ropt.components.executors import (
    HPCExecutor,
    ProcessExecutor,
    ThreadExecutor,
)
from ropt.exceptions import WorkflowError

from ._handlers import SharedHandlers, group_entries
from ._pool import WorkerPool

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from ropt.components.event_handlers import EventDispatcher, EventHandler
    from ropt.components.executors import Executor

    from ._report import ReportCallback

_STOPPED = "The background session is not running; open a new one."


class _Closable(Protocol):
    def close(self) -> None: ...


class _Session:
    """A background event loop and task group hosting pools and dispatchers.

    Whatever the session hands out is registered with it as an *extra*: any
    number may be open at once, and the session closes every one that is still
    open when it stops. That is what makes closing a session sufficient cleanup.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Two events, not one: `start` waits on `_ready` and must be released
        # however startup ends, while `_stopped` marks the loop as no longer
        # usable. Clearing `_ready` to mean that would deadlock `start`.
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._task_group: asyncio.TaskGroup | None = None
        self._shutdown: asyncio.Event | None = None
        self._failure: BaseException | None = None
        # Extras are added and closed from any thread: a driver thread may build
        # its own pool while the loop thread is closing the session down.
        self._extras_lock = threading.Lock()
        self._extras: list[_Closable] = []

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
        # A session that died carries its failure to whoever closes it, since
        # nothing else is watching the loop thread.
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
            # Keeps the loop serving until `stop` asks for shutdown. A loop that
            # has stopped but is not closed still accepts cross-thread work and
            # then never runs it, so a waiting caller would hang forever.
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
                # Retrieved so asyncio does not report them as never retrieved;
                # from a `finally`, because a task carrying a SystemExit would
                # otherwise abort the drain and skip this.
                for task in {main_task, *unfinished}:
                    if task.done() and not task.cancelled():
                        task.exception()

    async def _run(self) -> None:
        # The task group lives for as long as the session does: everything the
        # session hands out is started into it, which is why they all end when
        # the session does.
        self._shutdown = asyncio.Event()
        try:
            async with asyncio.TaskGroup() as task_group:
                self._task_group = task_group
                self._ready.set()
                await self._shutdown.wait()
                # Here, on the loop thread while the loop is still serving, so
                # the cancellations this schedules are actually run.
                self.close_extras()
        finally:
            # Set here, on the loop thread while the loop is still serving, so a
            # caller cannot pass the check in `_start_on_loop` and then hand work
            # to a loop that will not run it.
            self._stopped.set()

    def _start_on_loop(self, coro: Coroutine[Any, Any, None]) -> None:
        # Waits for the coroutine to finish, so what is handed back is already
        # running and a failure to start is raised at the factory call.
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

    def open_pool(
        self, make_executor: Callable[[], Executor], bundle_size: int = 1
    ) -> WorkerPool:
        """Start an executor on this session and wrap it in a pool.

        Args:
            make_executor: Builds the executor once the task group is known.
            bundle_size:   Evaluations per worker task, `0` for the whole batch.

        Returns:
            The pool, already running.
        """
        task_group = self._require_task_group()
        executor = make_executor()
        pool = WorkerPool(executor, self, bundle_size)
        # Registered before starting, so a session that shuts down during the
        # start still has the pool to cancel.
        self.add_extra(pool)
        try:
            self._start_on_loop(executor.start(task_group))
        except BaseException:
            self.discard_extra(pool)
            raise
        return pool

    def add_extra(self, extra: _Closable) -> None:
        with self._extras_lock:
            self._extras.append(extra)

    def discard_extra(self, extra: _Closable) -> None:
        with self._extras_lock:
            self._extras = [item for item in self._extras if item is not extra]

    def close_extras(self) -> None:
        # Take the list under the lock but close outside it: `close` calls back
        # into `discard_extra`, which takes the same lock.
        with self._extras_lock:
            extras, self._extras = self._extras, []
        for extra in extras:
            extra.close()

    def open_dispatcher(self, dispatcher: EventDispatcher) -> None:
        self._start_on_loop(dispatcher.start(self._require_task_group()))

    def close_dispatcher(self, dispatcher: EventDispatcher) -> None:
        schedule(self._loop, dispatcher.cancel)


class Session:
    """An open session, and the factories that build on it.

    A session owns one background event loop, on its own daemon thread. Pools
    need that loop to run on, which is why they are built here rather than
    constructed on their own. Everything a session hands out is returned to the
    caller and passed on explicitly; nothing is discovered from the surroundings.

    Bind the session with `as` and call its factories inside the block:

    ```python
    with session() as s:
        fast = s.thread_pool(workers=8)
        optimize(config, x0, function, pool=fast)
    ```

    Sessions are objects, so opening one inside another is unremarkable: each
    gets its own loop, and pools from different sessions never interact. A
    session is single use — once closed it cannot be reopened.
    """

    def __init__(self) -> None:
        """Initialize the session."""
        self._session: _Session | None = None
        self._entered = False

    def __enter__(self) -> Self:
        """Open the session's event loop.

        Returns:
            The session itself.

        Raises:
            WorkflowError: If the session was already opened.
        """
        if self._entered:
            msg = (
                "This session was already opened and cannot be entered again; "
                "open a separate session."
            )
            raise WorkflowError(msg)
        self._entered = True
        session = _Session()
        session.start()
        self._session = session
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the session, releasing every pool it created."""
        session, self._session = self._session, None
        if session is not None:
            session.stop()

    def thread_pool(self, *, workers: int = 1, bundle_size: int = 1) -> WorkerPool:
        """Create a pool that runs evaluations in worker threads.

        See [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            workers:     The number of worker threads.
            bundle_size: How many evaluations go to a worker as one task, `0`
                         for the whole batch. See
                         [`process_pool`][ropt.simple.Session.process_pool],
                         where it matters more.

        Returns:
            A pool backed by a thread pool.
        """
        return self._open_pool(lambda: ThreadExecutor(workers=workers), bundle_size)

    def process_pool(self, *, workers: int = 1, bundle_size: int = 1) -> WorkerPool:
        """Create a pool that runs evaluations in worker processes.

        The evaluation function must be picklable. See
        [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            workers:     The number of worker processes.
            bundle_size: How many evaluations go to a worker as one task. Every
                         task is transferred to a worker separately, and the
                         evaluations within one run after another, so this is a
                         trade between spreading a batch and the cost of moving
                         it. The default of 1 gives every evaluation its own
                         task, spreading a batch as widely as the workers allow;
                         a larger value groups that many per task; and `0` sends
                         the whole batch as a single task, which suits a pool
                         whose parallelism comes from the runs above it rather
                         than from within a batch.

        Returns:
            A pool backed by a process pool.
        """
        return self._open_pool(lambda: ProcessExecutor(workers=workers), bundle_size)

    def hpc_pool(  # ruff: ignore[too-many-arguments]
        self,
        *,
        workers: int = 1,
        cores: int = 1,
        cluster: str | None = None,
        queue: str | None = None,
        workdir: Path | str | None = None,
        config_path: Path | str | None = None,
        template: str | None = None,
        queue_type: str = "slurm",
        bundle_size: int = 1,
    ) -> WorkerPool:
        """Create a pool that runs evaluations on an HPC cluster.

        Interfaces with a cluster queue (for example Slurm) through `pysqa`; requires
        the `ropt[hpc]` extra, and the evaluation function must be picklable.
        The cluster is selected from `cluster`/`queue`: give a queue to search
        for its cluster, a cluster to use its default queue, or both to be
        explicit. See [Running Optimizations](../running/running.md) for a
        walkthrough.

        Args:
            workers:     The maximum number of concurrent cluster jobs.
            cores:       The number of CPUs per job.
            cluster:     The cluster name, when the `pysqa` config defines
                         several.
            queue:       The queue or partition name.
            workdir:     The shared-filesystem working directory (defaults to
                         the current directory).
            config_path: The path to the `pysqa` configuration directory.
            template:    An inline submission-script template, instead of a
                         config.
            queue_type:  The queueing system type.
            bundle_size: How many evaluations go to a worker as one task, `0`
                         for the whole batch. See
                         [`process_pool`][ropt.simple.Session.process_pool];
                         each task here is a cluster job.

        Returns:
            A pool backed by an HPC cluster.
        """
        resolved = Path.cwd() if workdir is None else Path(workdir).resolve()
        return self._open_pool(
            lambda: HPCExecutor(
                workers=workers,
                cores=cores,
                cluster=cluster,
                queue=queue,
                workdir=resolved,
                config_path=config_path,
                template=template,
                queue_type=queue_type,
            ),
            bundle_size,
        )

    def serial_pool(self) -> WorkerPool:
        """Create a pool that evaluates in-process, on the calling thread.

        Identical to the free [`serial_pool`][ropt.simple.serial_pool] function,
        except that this pool is closed when the session closes, so runs cannot
        keep using it afterwards. Prefer the free function for a pool that
        should outlive any session.

        Returns:
            A pool without an executor.
        """
        session = self._require_open()
        pool = WorkerPool(session=session)
        session.add_extra(pool)
        return pool

    def shared_handlers(
        self,
        *handler: EventHandler,
        threaded: EventHandler | Sequence[EventHandler] = (),
        report: ReportCallback | None = None,
    ) -> SharedHandlers:
        """Group result handlers that several runs share.

        Pass the group to every run that should feed it, in `handlers=`. Each
        handler then sees the results of all those runs, serialized across them,
        which is what makes accumulating over concurrent runs safe. A run may
        feed several groups, and mix them with handlers of its own.

        A handler joins one group at a time, and a handler that was ever passed
        to a run as a local handler cannot join a group at all; decide per
        handler whether it is local or shared. See
        [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            handler:  The result handlers to share, each run on the session's
                      event-loop thread.
            threaded: Handlers (one, or a sequence) to run on a worker thread
                      instead of the loop. This only helps handlers that spend
                      real time in blocking, GIL-releasing I/O (files,
                      databases, network); for in-memory work it gives no
                      benefit under CPython's GIL. See
                      [Running Optimizations](../running/running.md#running-a-handler-in-a-thread).
            report:   An optional callback invoked with an `EvaluateResult` for
                      each function evaluation across the group's runs.
                      Returning `True` stops the emitting run early with
                      `USER_ABORT` if it is an optimization; an evaluation has
                      no optimizer loop to interrupt, so there the return value
                      is ignored.

        Returns:
            A [`SharedHandlers`][ropt.simple.SharedHandlers] group.
        """
        return SharedHandlers(
            group_entries(handler, threaded, report), self._require_open()
        )

    def _open_pool(
        self, make_executor: Callable[[], Executor], bundle_size: int
    ) -> WorkerPool:
        return self._require_open().open_pool(make_executor, bundle_size)

    def _require_open(self) -> _Session:
        if self._session is None:
            msg = (
                "This session is not open; build pools inside its `with` block, "
                "for example `with session() as s: pool = s.thread_pool()`."
            )
            raise WorkflowError(msg)
        return self._session


def session() -> Session:
    """Open a background session that pools and shared handlers run on.

    The session owns one event loop, on a daemon thread, for as long as the
    block is open. Build pools on it with its factories, and pass them to the
    runs that should use them:

    ```python
    with session() as s:
        fast = s.thread_pool(workers=8)
        optimize(config, x0, function, pool=fast)
    ```

    Closing the session releases every pool it created, so most code needs no
    further cleanup. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Returns:
        A context manager owning the session, which binds the
        [`Session`][ropt.simple.Session] itself when used with `as`.
    """
    return Session()
