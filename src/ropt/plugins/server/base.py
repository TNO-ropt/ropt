"""Defines base classes for asynchronous servers."""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import queue


EvaluationResultType = TypeVar("EvaluationResultType")
ResultType = TypeVar("ResultType", bound="TaskResult")
TaskType = TypeVar("TaskType", bound="Task[Any, Any]")


@dataclass(frozen=True, kw_only=True)
class TaskResult:
    """A result from an asynchronous task.

    Attributes:
        task_id: The unique identifier of the task.
        result:  The result of the task.
    """

    task_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class Task(ABC, Generic[EvaluationResultType, ResultType]):
    """An asynchronous task to be executed by a worker.

    Attributes:
        result_queue: The queue to put the result in.
        task_id:      The unique identifier of the task.
    """

    result_queue: queue.Queue[ResultType]
    task_id: uuid.UUID = field(default_factory=uuid.uuid4)

    @abstractmethod
    async def run(self) -> EvaluationResultType:
        """Run the task.

        Returns:
            The result of the task.
        """


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
    ) -> Server[TaskType]:
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


class Server(ABC, Generic[TaskType]):
    """Abstract base class for server components within an optimization workflow.

    Subclasses must implement the following abstract methods and properties:

    - [`start`][ropt.plugins.server.base.Server.start]: Starts the server.
    - [`stop`][ropt.plugins.server.base.Server.stop]: Stops the server.
    - [`task_queue`][ropt.plugins.server.base.Server.task_queue]: Retrieves the
      servers task queue.
    """

    @property
    @abstractmethod
    def task_queue(self) -> asyncio.Queue[TaskType]:
        """Get the task queue."""

    @property
    @abstractmethod
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the asyncio loop used by this server."""

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
    def stop(self) -> None:
        """Stop the evaluation server."""


@dataclass(frozen=True, kw_only=True)
class EvaluatorTaskResult(TaskResult):
    """A result from an asynchronous task.

    Attributes:
        value:  The result of the task.
    """

    value: float


@dataclass(frozen=True, kw_only=True)
class EvaluatorTask(Task[float, EvaluatorTaskResult]):
    """An asynchronous task to be executed by a worker."""

    @abstractmethod
    async def run(self) -> float:
        """Run the task.

        Returns:
            The result of the evaluation.
        """


class EvaluatorServer(Server[EvaluatorTask]):
    """An server that performs evaluations."""

    def __init__(self) -> None:
        """Initialize the DefaultAsyncEvaluator."""
        super().__init__()
        self._task_queue: asyncio.Queue[EvaluatorTask] = asyncio.Queue()

    @property
    def task_queue(self) -> asyncio.Queue[EvaluatorTask]:
        """Get the task queue."""
        return self._task_queue
