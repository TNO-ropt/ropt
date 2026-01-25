"""Defines base classes for asynchronous servers."""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import queue
    from collections.abc import Callable


T = TypeVar("T", bound="Task[Any, Any]")
R = TypeVar("R")
TR = TypeVar("TR")


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
            name:    The requested server name (potentially plugin-specific).
            kwargs:  Additional arguments for custom configuration.

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
    def start(self, task_group: asyncio.TaskGroup) -> None:
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


class ServerBase(Server, Generic[T]):
    """An base class for asynchronous servers."""

    def __init__(self, maxsize: int = 0) -> None:
        """Initialize the server.

        Arguments:
            maxsize: Maximum size of the task queue.
        """
        super().__init__()
        self._task_queue: asyncio.Queue[T] = asyncio.Queue(maxsize)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._stopped = threading.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the asyncio loop used by this server."""
        return self._loop

    @property
    def task_group(self) -> asyncio.TaskGroup | None:
        """Get the task group used by this server."""
        return self._task_group

    @property
    def task_queue(self) -> asyncio.Queue[T]:
        """Get the task queue."""
        return self._task_queue

    def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the evaluation server.

        This method should be overload to implement starting the server. To
        populate the loop and task_group properties, and to implement a check
        that the server is not already running, this method can be called in the
        overloaded function.

        Args:
            task_group: The task group to use.

        Raises:
            RuntimeError: If the evaluator is already running or using an
                          external queue.
        """
        if not self._stopped:
            msg = "Server is already running."
            raise RuntimeError(msg)
        self._stopped.clear()
        self._loop = asyncio.get_running_loop()
        self._task_group = task_group

    def cancel(self) -> None:
        """Stop the evaluation server."""
        self._stopped.set()

    def is_running(self) -> bool:
        """Check if the server is not running.

        Returns:
            True if the server is not running.
        """
        return not self._stopped.is_set()


@dataclass(frozen=True, kw_only=True)
class TaskResult:
    """A result from a task."""


@dataclass(frozen=True, kw_only=True)
class Task(ABC, Generic[R, TR]):
    """A task to be executed by a worker.

    Attributes:
        function:     The function to execute.
        result_queue: The queue to put the result in.
    """

    function: Callable[[], R]
    result_queue: queue.Queue[TR]

    @abstractmethod
    def put_result(self, result: R) -> None:
        """Put the result in the result queue."""

    @abstractmethod
    def put_exception(self, exc: Exception) -> None:
        """Put an exception in the result queue."""
