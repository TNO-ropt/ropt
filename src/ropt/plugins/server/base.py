"""Defines base classes for asynchronous servers."""

from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from collections.abc import Callable


class ServerPlugin(Plugin):
    """Abstract base class for asynchronous server plugins.

    This class defines the interface for plugins responsible for creating
    [`Evaluator`][ropt.plugins.server.base.Server] instances within
    an optimization workflow.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> Server:
        """Create a Server instance.

        This abstract class method serves as a factory for creating concrete
        [`Server`][ropt.plugins.server.base.Server] objects. Plugin
        implementations must override this method to return an instance of their
        specific `Server` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        an evaluator provided by this plugin is requested.

        The `name` argument specifies the requested evaluator, potentially
        in the format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created evaluator if
        the plugin supports multiple evaluator types.

        Args:
            name:   The requested server name (potentially plugin-specific).
            kwargs: Additional arguments for custom configuration.

        Returns:
            An initialized instance of an `Server` subclass.
        """


class Server(ABC):
    """Abstract base class for server components within an optimization workflow.

    Subclasses must implement the following abstract methods and properties:

    - [`start`][ropt.plugins.server.base.Server.start]: Starts the server.
    - [`cancel`][ropt.plugins.server.base.Server.cancel]: Stops the server.
    - [`task_queue`][ropt.plugins.server.base.Server.task_queue]: Retrieves the
      servers task queue.
    - [`loop`][ropt.plugins.server.base.Server.loop]: Retrieves the
      currently running asyncio loop.
    - [`task_group`][ropt.plugins.server.base.Server.task_group]: The asyncio.Taskgroup
      used by this server.
    - [`is_running`][ropt.plugins.server.base.Server.is_running]: Checks if the
      server is running.
    """

    @property
    @abstractmethod
    def task_queue(self) -> asyncio.Queue[Any]:
        """Get the task queue."""

    @property
    @abstractmethod
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the asyncio loop used by this server."""

    @property
    @abstractmethod
    def task_group(self) -> asyncio.TaskGroup | None:
        """Get the task group used by this server."""

    @abstractmethod
    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the evaluation server.

        Args:
            task_group: The task group to use.

        Raises:
            RuntimeError: If the evaluator is already running or using an
                          external queue.
        """

    @abstractmethod
    def cancel(self) -> None:
        """Stop the evaluation server."""

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the server is not running.

        Returns:
            True if the server is not running.
        """


class ServerBase(Server):
    """An base class for asynchronous servers."""

    def __init__(self, queue_size: int = 0) -> None:
        """Initialize the server.

        Arguments:
            queue_size: Maximum size of the task queue.
        """
        super().__init__()
        self._task_queue: asyncio.Queue[Task] = asyncio.Queue(queue_size)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._running = threading.Event()
        self._ready_event = asyncio.Event()
        self._wait_event = asyncio.Event()
        self._wait_task: asyncio.Task[None] | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the asyncio loop used by this server."""
        return self._loop

    @property
    def task_group(self) -> asyncio.TaskGroup | None:
        """Get the task group used by this server."""
        return self._task_group

    @property
    def task_queue(self) -> asyncio.Queue[Task]:
        """Get the task queue."""
        return self._task_queue

    async def _finish_start(self, task_group: asyncio.TaskGroup) -> None:
        if self._running.is_set():
            msg = "Server is already running."
            raise RuntimeError(msg)
        self._running.set()
        self._loop = asyncio.get_running_loop()
        self._task_group = task_group
        self._ready_event.clear()
        self._wait_event.clear()
        self._wait_task = task_group.create_task(self._wait_for_cancel())
        await self._ready_event.wait()

    async def _wait_for_cancel(self) -> None:
        self._ready_event.set()
        try:
            await self._wait_event.wait()
        finally:
            if self._running.is_set():
                self._running.clear()
                self.cleanup()

    def cancel(self) -> None:
        """Stop the evaluation server."""
        if self._wait_task is not None:
            self._wait_task.cancel()
            self._wait_task = None

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the server."""

    def is_running(self) -> bool:
        """Check if the server is not running.

        Returns:
            True if the server is running.
        """
        return self._running.is_set()

    def _drain_and_kill(self) -> None:
        """Drain the task queue and kill clients."""
        while not self._task_queue.empty():
            try:
                task = self._task_queue.get_nowait()
                task.cancel_all()
                self._task_queue.task_done()
            except asyncio.QueueEmpty:
                break


@dataclass(kw_only=True)
class Task(ABC):
    """A task to be executed by a worker.

    Attributes:
        id:            A unique identifier for the task (set on construction).
        function:      The function to execute.
        args:          The arguments to pass to the function.
        kwargs:        The keyword arguments to pass to the function.
        results_queue: The queue to put the result in.
        result:        The result of the function, or None if no result is available.
    """

    id: UUID = field(default_factory=uuid4)
    function: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    results_queue: ResultsQueue
    result: Any | None = None

    def put_result(self, result: Any) -> None:  # noqa: ANN401
        """Put the result in the result queue."""
        self.result = result
        self.results_queue.put(self)

    def cancel_all(self) -> None:
        """Stop putting results in the result queue."""
        self.results_queue.put(None)
        self.results_queue.close()


class ResultsQueue(queue.Queue[Task | None]):
    """A queue that can be closed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the queue."""
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self) -> None:
        """Close the queue."""
        self.closed = True

    def put(self, item: Task | None, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Put an item in the queue."""
        if not self.closed:
            super().put(item, *args, **kwargs)
