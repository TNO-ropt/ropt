import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Self

from ._events import ErrorEvent, Event
from ._task import TaskBase


class PollerTask(ABC):
    """A background task that periodically polls a set of tasks for completion.

    Subclasses must implement the `poll` method. This method should check the
    status of the managed tasks and call `mark_finished()` on any that have
    completed.

    Note:
        The `add()` method does not validate tasks. It is the caller's
        responsibility to ensure that only "pollable" tasks (those whose
        completion is determined externally) are added to the poller. Tasks that
        call `mark_finished()` themselves should not be polled.
    """

    def __init__(
        self,
        *,
        event_queue: asyncio.Queue[Event] | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        """Initializes the PollerTask.

        Args:
            event_queue:   An optional queue for unhandled exceptions.
            poll_interval: The interval in seconds between polls.

        Note:
            The `event_queue` is used to report unhandled exceptions that occur
            within the poller's `poll` method, not for exceptions raised by the
            tasks being polled.
        """
        self._event_queue = event_queue
        self._poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._tasks: set[TaskBase] = set()
        self._cancelled = False

    def add(self, tasks: TaskBase | Iterable[TaskBase]) -> None:
        """Adds a task, or a list of tasks to be polled.

        Args:
            tasks: The task(s) to add.
        """
        if isinstance(tasks, Iterable):
            self._tasks |= set(tasks)
        else:
            self._tasks.add(tasks)

    def schedule(self) -> Self:
        """Starts the poller's background task.

        Returns:
            The instance of the poller task.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
            self._task.add_done_callback(self._done_callback)
        return self

    @abstractmethod
    def poll(self, tasks: set[TaskBase]) -> None:
        """Abstract method to be implemented by subclasses."""

    async def cancel(self) -> None:
        """Stops the poller and clears the list of tasks."""
        self._cancelled = True
        self._tasks.clear()
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def _done_callback(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return

        exc = task.exception()
        if exc is not None:
            self._report_error(
                exc, "Unhandled task exception could not be queued by the poller"
            )

    async def _run(self) -> None:
        try:
            while not self._cancelled:
                if self._tasks:
                    try:
                        self.poll(self._tasks)
                    except Exception as exc:  # noqa: BLE001
                        self._report_error(exc, "Unhandled exception in poller")
                    self._tasks -= {
                        t for t in self._tasks if t.state in {"done", "cancelled"}
                    }
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            pass

    def _report_error(
        self,
        exc: BaseException,
        message: str,
    ) -> None:
        def _error_fallback(exc: BaseException) -> None:
            asyncio.get_running_loop().call_exception_handler(
                {"message": message, "exception": exc}
            )

        if self._event_queue is None:
            _error_fallback(exc)
        else:
            try:
                self._event_queue.put_nowait(ErrorEvent(source="poller", error=exc))
            except Exception:  # noqa: BLE001
                _error_fallback(exc)
