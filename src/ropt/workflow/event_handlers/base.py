"""Base classes for event handler plugins and event handlers."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent


class EventHandler(ABC):
    """Abstract base class for event handlers.

    This class defines the fundamental interface for all event handlers within
    an optimization workflow. Concrete handler implementations, (e.g., tracking
    results, storing data, logging), must inherit from this base class.

    Handlers may store state using dictionary-like access (`[]`), allowing
    them to accumulate information or make data available to other components in
    an optimization workflow.

    Subclasses must implement the abstract
    [`handle_event`][ropt.workflow.event_handlers.EventHandler.handle_event]
    method to define their specific event processing logic.

    Event handlers are attached to a
    [`ComputeStep`][ropt.workflow.compute_steps.ComputeStep] using its
    [`add_event_handler`][ropt.workflow.compute_steps.ComputeStep.add_event_handler]
    method. When the compute step emits an event, the `handle_event` method of
    each attached handler is invoked, allowing it to process the event.

    Thread safety:
        Every handler owns a re-entrant lock created at construction. The
        base-class `__getitem__` and `__setitem__` acquire it, so
        dictionary-style access is safe across threads. Subclasses (and
        external callers that need atomic compound operations) should wrap
        critical sections with
        [`locked()`][ropt.workflow.event_handlers.EventHandler.locked]:

            with handler.locked():
                snapshot = handler["results"]
                count = len(snapshot)

        The lock is re-entrant, so user callbacks invoked from within a
        locked region may safely call back into the handler on the same
        thread.
    """

    def __init__(self) -> None:
        """Initialize the EventHandler."""
        self.__stored_values: dict[str, Any] = {}
        # Re-entrant: subclasses commonly invoke user callbacks while holding
        # the lock, and those callbacks may call back into the handler (e.g.
        # `handler[key]`) on the same thread.
        self.__lock = threading.RLock()

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state, omitting the non-picklable lock."""
        # threading.RLock is not picklable; drop it and recreate in __setstate__.
        state = self.__dict__.copy()
        state.pop("_EventHandler__lock", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and recreate the lock dropped by `__getstate__`."""
        self.__dict__.update(state)
        self.__lock = threading.RLock()

    def locked(self) -> AbstractContextManager[None]:
        """Return a context manager that holds the handler's internal lock.

        Use this to wrap any compound operation that must observe a
        consistent view of the handler's state:

            with handler.locked():
                snapshot = handler["results"]
                count = len(snapshot)

        Individual `handler[key]` reads and `handler[key] = value` writes
        are already serialized internally, so explicit locking is only
        needed when multiple accesses must be atomic with respect to each
        other.

        The lock is re-entrant: entering it from a thread that already
        holds it (e.g. from a user callback dispatched from within
        `handle_event`) is safe and will not deadlock.

        Returns:
            A context manager that acquires the lock on entry and releases
            it on exit.
        """
        return self.__locked_cm()

    @contextmanager
    def __locked_cm(self) -> Iterator[None]:
        with self.__lock:
            yield

    @property
    @abstractmethod
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """

    @abstractmethod
    def handle_event(self, event: EnOptEvent) -> None:
        """Process an event.

        This abstract method must be implemented by concrete `EventHandler`
        subclasses. It defines the event handler's core logic for reacting to
        [`EnOptEvent`][ropt.events.EnOptEvent] objects emitted in the
        optimization workflow.

        Implementations should inspect the `event` object (its `event_type` and
        `data`) and perform computations accordingly, such as storing results,
        logging information, or updating internal state.

        Args:
            event: The event object.
        """

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Retrieve a value from the event handler's internal state.

        This method enables dictionary-like access (`handler[key]`) to the
        values stored within the event handler's internal state dictionary. This
        allows handlers to store and retrieve data accumulated during workflow
        execution.

        Args:
            key: The string key identifying the value to retrieve.

        Returns:
            The value associated with the specified key.

        Raises:
            AttributeError: If the provided `key` does not exist in the
                            event handler's stored values.
        """
        with self.__lock:
            if key in self.__stored_values:
                return self.__stored_values[key]
            msg = f"Unknown event handler data key: `{key}`"
            raise AttributeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Store or update a value in the event handler's internal state.

        This method enables dictionary-like assignment (`handler[key] = value`)
        to store arbitrary data within the event handler's internal state
        dictionary. This allows event handlers to accumulate information or make
        data available to other components of the workflow.

        The key must be a valid Python identifier.

        Args:
            key:   The string key identifying the value to store (must be an identifier).
            value: The value to associate with the key.

        Raises:
            AttributeError: If the provided `key` is not a valid identifier.
        """
        if not key.isidentifier():
            msg = f"Not a valid key: `{key}`"
            raise AttributeError(msg)
        with self.__lock:
            self.__stored_values[key] = value
