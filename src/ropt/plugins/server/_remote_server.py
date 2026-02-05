"""Defines base classes for asynchronous servers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from ropt.enums import ExitCode
from ropt.exceptions import ComputeStepAborted

from .base import RemoteTaskState, ServerBase, Task

if TYPE_CHECKING:
    import queue
    from uuid import UUID


T = TypeVar("T", bound="Task[Any, Any]")
R = TypeVar("R")
TR = TypeVar("TR")


class RemoteAdapter(Protocol, Generic[R, TR]):
    """A protocol for remote servers.

    This adapter class is used by the
    [`RemoteServer`][ropt.plugins.server._remote_server.RemoteServer] class to
    submit and monitor tasks running on a remote server, and retrieve their
    results. A class that adheres to this protocol must implement the following
    methods:

    - [`submit`][ropt.plugins.server._remote_server.RemoteAdapter.submit]:
      Submit a task to the server.
    - [`poll`][ropt.plugins.server._remote_server.RemoteAdapter.poll]: Poll the
      server for tasks.
    - [`get_result`][ropt.plugins.server._remote_server.RemoteAdapter.get_result]:
      Get the result of a task.

    """

    def submit(self, task: Task[R, TR]) -> None:
        """Submit a task to the server.

        Args:
            task: The task to submit.
        """

    def poll(self) -> dict[UUID, RemoteTaskState]:
        """Poll the server for task state changes."""

    def get_result(self, task_id: UUID) -> R | Exception:
        """Get the result of a task.

        Args:
            task_id: The id of the task to get the result of.

        Returns:
            The result of the task, or an except that may have been raised.
        """


class DefaultRemoteServer(ServerBase[Task[R, TR]]):
    """A class for servers that run tasks remotely."""

    def __init__(
        self,
        *,
        remote: RemoteAdapter[R, TR],
        workers: int = 1,
        maxsize: int = 0,
        interval: float = 1,
    ) -> None:
        """Initialize the server.

        Args:
            remote:   The adapter class for the remote server.
            workers:  The number of workers to use.
            maxsize:  Maximum size of the tasks queue.
            interval: Polling interval in seconds.
        """
        super().__init__(maxsize=maxsize)
        self._remote = remote
        self._workers = workers
        self._interval = interval
        self._worker_task: asyncio.Task[None] | None = None
        self._tasks: dict[UUID, tuple[Task[R, TR], RemoteTaskState]] = {}

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the server.

        Args:
            task_group: The task group to use.
        """
        self._worker_task = task_group.create_task(self._worker())
        await self._finish_start(task_group)

    async def _worker(self) -> None:
        while self._running.is_set():
            remove = []
            for task_id, (task, state) in self._tasks.items():
                result = self._remote.get_result(task_id)
                if state == "done":
                    assert not isinstance(result, Exception)
                    task.put_result(result)
                if state == "error":
                    assert isinstance(result, Exception)
                    task.put_result(result)
                remove.append(task_id)
            for task_id in remove:
                self._tasks.pop(task_id)
            if len(self._tasks) < self._workers:
                try:
                    task = self._task_queue.get_nowait()
                    self._tasks[task.id] = (task, "pending")
                    self._remote.submit(task)
                    self._task_queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            for task_id, state in self._remote.poll().items():
                if task_id in self._tasks:
                    task, _ = self._tasks[task_id]
                    self._tasks[task_id] = (task, state)
            await asyncio.sleep(self._interval)

    def cleanup(self) -> None:
        """Clean up the server."""
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        self._worker_task = None
        queues: set[queue.Queue[TR]] = set()
        for task, _ in self._tasks.values():
            if task.result_queue not in queues:
                queues.add(task.result_queue)
                task.put_result(ComputeStepAborted(ExitCode.ABORT_FROM_ERROR))
        self._drain_and_kill(ComputeStepAborted(ExitCode.ABORT_FROM_ERROR))
