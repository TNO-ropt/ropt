"""This module implements the event server."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ropt.events import EnOptEvent
    from ropt.workflow.event_handlers import EventHandler


class EventServer:
    """An event server that dispatches events to handlers from the asyncio event loop.

    See [Parallel Evaluation](../usage/parallel.md#event-server) for usage.
    """

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []
        self._queue: asyncio.Queue[EnOptEvent | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = threading.Event()

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add an event handler.

        Args:
            handler: The handler to add.
        """
        self._handlers.append(handler)

    def put_event(self, event: EnOptEvent) -> None:
        """Submit an event from any thread.

        Args:
            event: The event to submit.
        """
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def is_running(self) -> bool:
        """Check if the server is running.

        Returns:
            True if the server is running.
        """
        return self._running.is_set()

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the event server.

        Args:
            task_group: The task group to use.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._running.is_set():
            msg = "EventServer is already running."
            raise RuntimeError(msg)
        self._queue = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._running.set()
        task_group.create_task(self._process())

    def cancel(self) -> None:
        """Stop the event server."""
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def _dispatch(self, event: EnOptEvent) -> None:
        for handler in self._handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)

    def _drain(self) -> None:
        assert self._queue is not None
        while not self._queue.empty():
            remaining = self._queue.get_nowait()
            self._queue.task_done()
            if remaining is not None:
                self._dispatch(remaining)

    async def _process(self) -> None:
        assert self._queue is not None
        try:
            while True:
                event = await self._queue.get()
                self._queue.task_done()
                if event is None:
                    self._drain()
                    break
                self._dispatch(event)
        finally:
            self._running.clear()
