"""Base classes for event handler plugins and event handlers.

`_event_lock` covers one call to `handle_event`, and makes a second one from
another thread wait. `_event_owner` records the thread inside it, so that a
re-entrant call raises instead of deadlocking on a lock it already holds.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent


class EventHandler(ABC):
    """Abstract base class for event handlers.

    A concrete handler reacts to the events emitted by a
    [`ComputeStep`][ropt.components.compute_steps.ComputeStep] it is attached to,
    by implementing
    [`handle_event`][ropt.components.event_handlers.EventHandler.handle_event].
    Handlers may store state using dictionary-like access (`[]`).

    Note:
        A handler's `handle_event` serializes itself: a call from a second
        thread waits for the first to finish, so the same handler may be
        attached to several compute steps running at once. It is not
        re-entrant, so a call that reaches the same handler again on the same
        stack raises `WorkflowError` rather than deadlocking. See
        [Optimization Workflows](../advanced/workflows.md#event-handlers) for
        usage and pitfalls.
    """

    def __init__(self) -> None:
        """Initialize the EventHandler."""
        # Name-mangled, so a subclass cannot reach it by accident and the `[]`
        # access stays the only way in.
        self.__stored_values: dict[str, Any] = {}
        self._event_lock = threading.Lock()
        self._event_owner: int | None = None

    @property
    @abstractmethod
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """

    @abstractmethod
    def _handle_event(self, event: EnOptEvent) -> None:
        """React to an emitted event.

        Implemented by concrete subclasses; callers use `handle_event`, which
        adds the concurrency guard.

        Args:
            event: The event object.
        """

    def handle_event(self, event: EnOptEvent) -> None:
        """React to an emitted event.

        Calls from other threads are serialized: the second waits for the first
        to finish.

        Args:
            event: The event object.

        Raises:
            WorkflowError: If this handler is already running on this call stack.
        """
        # Read before acquiring, without synchronization: the only owner id a
        # thread can read that equals its own is one it wrote itself.
        if self._event_owner == threading.get_ident():
            msg = "This event handler is already running further up this call stack."
            raise WorkflowError(msg)
        with self._event_lock:
            self._event_owner = threading.get_ident()
            try:
                self._handle_event(event)
            finally:
                self._event_owner = None

    def __getitem__(self, key: str) -> Any:  # ruff: ignore[any-type]
        """Retrieve a stored value by key (`handler[key]`).

        Args:
            key: The string key identifying the value to retrieve.

        Returns:
            The value associated with the specified key.

        Raises:
            AttributeError: If `key` does not exist in the stored values.
        """
        if key in self.__stored_values:
            return self.__stored_values[key]
        msg = f"Unknown event handler data key: `{key}`"
        raise AttributeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:  # ruff: ignore[any-type]
        """Store or update a value in the internal state (`handler[key] = value`).

        Args:
            key:   The string key identifying the value to store (must be an identifier).
            value: The value to associate with the key.

        Raises:
            AttributeError: If `key` is not a valid identifier.
        """
        if not key.isidentifier():
            msg = f"Not a valid key: `{key}`"
            raise AttributeError(msg)
        self.__stored_values[key] = value
