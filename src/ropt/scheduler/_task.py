import asyncio
import contextlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Self, TypeAlias

from ._events import ErrorEvent, Event

State: TypeAlias = Literal["pending", "running", "done", "cancelled"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskEvent(ErrorEvent):
    """Event reporting on the state of a task.

    The inherited `source` and `error` fields are set to the task ID (`tid`) of
    the tasks, and to any errors that may have occurred.

    Attributes:
        state: The current state of the task.
    """

    state: State


class TaskBase(ABC):
    """Abstract base class for a task.

    This class provides the basic infrastructure for a task that can be
    scheduled, executed, and have its state monitored. It manages task
    dependencies, lifecycle state, and state reporting.

    Subclasses must implement the `run` method to define the task's core logic.
    The task is scheduled for asynchronous execution via the `schedule` method.
    Task completion can be signaled in two primary ways:

    1.  **Self-completing tasks**: The `run` method executes all necessary
        functionality and then explicitly calls `mark_finished()` to signal
        completion.
    2.  **Externally-monitored tasks**: The `run` method initiates an
        independent process. An external component, upon detecting the
        completion of this process, calls `mark_finished()` on the task.

    The state the task can be inspected via the `state` property which can have
    one of the following values: "pending", "running", "done", or "cancelled".
    Changes in the state, and errors are emitted as `TaskEvent` objects, via an
    `asyncio.Queue` that is passed to the `schedule` method.

    The lifecycle of a task is as follows: 1.  **Initialization**: The task
    starts in the "pending" state. 2.  **Scheduling**: `schedule()` is called,
    which creates an
        `asyncio.Task` to execute the `run` method. A `TaskEvent` with state
        "running" is emitted.
    3.  **Completion**: `mark_finished()` is called, either by the task itself
        or an external monitor. A `TaskEvent` with state "done" is emitted.
    4.  **Cancellation**: `cancel()` can be called before the task is finished.
        A `TaskEvent` with state "cancelled" is emitted.
    5.  **Error Handling**: If an exception occurs during execution, the final
        `TaskEvent` ("done" or "cancelled") will include the exception in its
        `error` field. - If the exception occurs within the `run` method, the
        task's state
          transitions to "cancelled" if the task is being cancelled, or "done"
          otherwise.
        - If the exception occurs during cancellation (e.g., within a `finally`
          block), the task's state transitions to "cancelled", and the exception
          is attached to the `TaskEvent`.

    The `schedule` method accepts an optional `asyncio.Semaphore` object, which
    can be used to limit the number of concurrent running tasks.

    The task object maintains a list of task IDs (`dependencies` property) that
    indicate what other tasks must complete before this task can be started.
    This merely lists the dependencies, the actual dependency logic is not
    implemented by the `TaskBase` class itself.

    Note: Error handling.
        Exceptions raised inside the `run` method are folded into the final
        `TaskEvent`, whose state will be either "done" or "cancelled". Consumers
        should inspect the `error` field of this final event to determine
        whether the task failed. Exceptions that originate within the internals
        of `TaskBase` itself are emitted separately as a plain `ErrorEvent`.
    """

    def __init__(
        self,
        *,
        tid: uuid.UUID | None = None,
        dependencies: list[uuid.UUID] | None = None,
    ) -> None:
        """Initializes the TaskBase.

        If `tid` is None, a new UUID is generated.

        Args:
            tid:          The unique identifier of the task.
            dependencies: A list of task IDs that this task depends on.
        """
        self._tid = tid or uuid.uuid4()
        self._dependencies: list[uuid.UUID] = dependencies or []
        self._event_queue: asyncio.Queue[Event] | None = None
        self._state: State = "pending"
        self._task: asyncio.Task[None] | None = None
        self._finished = asyncio.Event()
        self._cancelling = False

    @property
    def tid(self) -> uuid.UUID:
        """The unique identifier of the task."""
        return self._tid

    @property
    def dependencies(self) -> list[uuid.UUID]:
        """A list of task IDs that this task depends on."""
        return self._dependencies

    @property
    def state(self) -> State:
        """The current state of the task."""
        return self._state

    def schedule(
        self,
        event_queue: asyncio.Queue[Event],
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> Self:
        """Schedules the task for execution.

        This creates an `asyncio.Task` to run the task. If a semaphore is
        provided, it will be used to limit the number of concurrently running
        tasks.

        Args:
            event_queue: The queue for reporting task state updates and errors.
            semaphore:   Optional semaphore to limit concurrency.

        Returns:
            The instance of the task.
        """
        if self._task is not None:
            return self
        self._event_queue = event_queue
        if semaphore is None:
            self._task = asyncio.create_task(self._execute())
        else:
            self._task = asyncio.create_task(self._execute_with_semaphore(semaphore))
        self._task.add_done_callback(self._done_callback)
        return self

    @abstractmethod
    async def run(self) -> None:
        """The main entry point for the task's execution logic.

        For self-completing tasks, this method should perform all its work and
        then call `mark_finished()`. For externally-monitored tasks, this method
        should only start a background process; `mark_finished()` will be called
        by an external monitor.
        """

    def mark_finished(self) -> None:
        """Marks the task as finished.

        This method signals that the task's work is complete. It should be
        called either by the `run` method upon its completion (for
        self-completing tasks) or by an external monitor when a background
        process initiated by `run` has finished.
        """
        self._finished.set()

    async def cancel(self) -> None:
        """Cancels the task.

        This will attempt to cancel the underlying `asyncio.Task`.
        """
        if self._task and not self._task.done():
            if self._task.cancel():
                self._cancelling = True
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _done_callback(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return

        def _error_fallback(exc: BaseException) -> None:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Unhandled task exception could not be queued",
                    "exception": exc,
                    "task": task,
                }
            )

        exc = task.exception()
        if exc is not None:
            if self._event_queue is None:
                _error_fallback(exc)
            else:
                try:
                    self._event_queue.put_nowait(
                        ErrorEvent(source=self._tid, error=exc)
                    )
                except Exception:  # noqa: BLE001
                    _error_fallback(exc)

    async def _set_status(
        self, /, state: State | None = None, error: BaseException | None = None
    ) -> None:
        if state is not None:
            self._state = state
        assert self._event_queue is not None
        kwargs: dict[str, Any] = {"source": self._tid}
        if state is not None:
            kwargs["state"] = state
        if error is not None:
            kwargs["error"] = error
        await self._event_queue.put(TaskEvent(**kwargs))

    async def _execute_with_semaphore(self, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            await self._execute()

    async def _execute(self) -> None:
        try:
            await self._set_status(state="running")
            await self.run()
            await self._finished.wait()
            await self._set_status(state="done")
        except asyncio.CancelledError:
            await self._set_status(state="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            await self._set_status(
                state="cancelled" if self._cancelling else "done", error=exc
            )
