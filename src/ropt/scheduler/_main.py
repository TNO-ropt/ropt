import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from ._events import ErrorEvent, Event
from ._poller import PollerTask
from ._scheduler import Scheduler
from ._task import TaskBase


@dataclass(frozen=True, slots=True, kw_only=True)
class StopEvent(Event):
    """Sentinel class for stopping a main loop."""


class MainEventLoop(ABC):
    """Provides a customizable main event loop for running and managing tasks.

    This class orchestrates the task execution process by integrating a
    `Scheduler` for dependency management and an optional `Poller` for
    monitoring externally-managed tasks. It operates on a central event queue,
    processing task status updates and errors.

    **Usage:**
    1.  Subclass `MainEventLoop` and implement the `handle_event` method to
        define application-specific logic, such as tracking progress or stopping
        the loop when all tasks are complete.
    2.  Optionally, override `handle_error` for custom error handling. The
        default behavior is to re-raise any received exception.
    3.  Instantiate your subclass.
    4.  Add tasks using the `add_tasks` method.
    5.  Call and `await` the `run()` method to start the loop.
    6.  While awaiting the `run()` method, you may add additional tasks from within
        the `handle_event` method.

    **Methods:**
    - `add_tasks()`: Submits tasks to the `Scheduler` and, if configured, to
      the `Poller`. Can be called before or during the loop's execution.
    - `run()`: Starts processing events from the internal event queue.
    - `stop()`: Places a `StopEvent` on the queue to gracefully terminate the
      `run()` method.
    - `cleanup()`: Called automatically when `run()` exits. It ensures the
      shutdown of the `Scheduler` and `Poller`. This method can be extended, but
      subclasses should call `super().cleanup()`.
    """

    def __init__(
        self,
        *,
        max_tasks: int | None = None,
        poller: PollerTask | None = None,
    ) -> None:
        """Initializes the MainEventLoop.

        Sets up the internal queues and initializes the `Scheduler` component.
        If `max_tasks` is provided, a semaphore is used to limit the number of
        tasks that can run simultaneously.

        Args:
            max_tasks: The maximum number of tasks that can run concurrently.
            poller:    Optional poller object.
        """
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = (
            asyncio.Queue()
        )
        self._scheduler = Scheduler(
            self._request_queue,
            event_queue=self._event_queue,
            semaphore=asyncio.Semaphore(max_tasks) if max_tasks else None,
        ).schedule()
        self._poller = poller.schedule() if poller else None

    async def add_tasks(self, tasks: TaskBase | Iterable[TaskBase]) -> None:
        """Adds one or more tasks to be scheduled.

        Tasks added here will be processed by the internal scheduler and
        optionally added to the poller.

        Args:
            tasks: A single tasks or an iterable of tasks to be added.
        """
        if self._poller is not None:
            self._poller.add(tasks)
        await self._request_queue.put(tasks)

    async def run(self) -> None:
        """Starts the main event loop.

        This method continuously retrieves events from the event queue and
        dispatches them to `handle_error` or `handle_event` until `stop()`
        is called.
        """
        try:
            while True:
                event = await self._event_queue.get()
                if isinstance(event, StopEvent):
                    break
                if isinstance(event, ErrorEvent) and event.error is not None:
                    await self.handle_error(event)
                else:
                    await self.handle_event(event)
        finally:
            await self.cleanup()

    async def handle_error(self, event: ErrorEvent) -> None:  # noqa: PLR6301
        """Handles `ErrorEvent` instances.

        By default, this method re-raises the exception contained within an
        `Event` that derives from `ErrorEvent`. Other event types are ignored.

        Subclasses can override this method to implement custom error handling
        logic.

        Note:
            Because this derives from `ErrorEvent` this will also raise errors
            stored in a task object. Override if this is not desired.

        Args:
            event: The event object to handle.
        """
        assert event.error is not None
        raise event.error

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Abstract method to handle general events.

        Subclasses must implement this method to define how non-error events
        are processed.

        Args:
            event: The `Event` object to handle.
        """

    def stop(self) -> None:
        """Stops the main event loop.

        This method puts a `StopEvent` into the event queue, which signals the
        main loop to exit gracefully.
        """
        self._event_queue.put_nowait(StopEvent(source="main_loop"))

    async def cleanup(self) -> None:
        """Performs cleanup operations when the main loop stops.

        This method cancels the background tasks and ensures a graceful shutdown
        of the task runner components.
        """
        if self._poller is not None:
            await self._poller.cancel()
        await self._scheduler.cancel()
