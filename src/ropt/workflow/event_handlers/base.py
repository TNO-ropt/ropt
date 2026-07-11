"""Base classes for event handler plugins and event handlers."""

from __future__ import annotations

import functools
import threading
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent


class _Attachment(Enum):
    """How an event handler is attached within a workflow."""

    NONE = auto()
    DISPATCHER = auto()
    COMPUTE_STEP = auto()


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

    Note:
        Event handlers are single-thread objects. A handler attached to compute
        steps binds to the thread that first calls `handle_event` and raises a
        `RuntimeError` if called from another thread. To receive events from
        multiple threads, register it with an
        [`EventDispatcher`][ropt.workflow.event_handlers.EventDispatcher], which
        serializes the calls. A handler may be owned by at most one dispatcher,
        or by one or more compute steps, but not both. See
        [Optimization Workflows](../usage/workflows.md#event-handlers) for usage
        and pitfalls.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)
        if "handle_event" in cls.__dict__ and not getattr(
            cls.__dict__["handle_event"], "__wrapped__", None
        ):
            original = cls.__dict__["handle_event"]

            @functools.wraps(original)
            def _guarded(
                self: EventHandler,
                event: EnOptEvent,
                *,
                _orig: Any = original,  # noqa: ANN401
            ) -> None:
                if self._attached_to is not _Attachment.DISPATCHER:
                    current = threading.get_ident()
                    with self._owner_lock:
                        if self._owner_thread is None:
                            self._owner_thread = current
                        elif self._owner_thread != current:
                            msg = "This event handler cannot be used from more than one thread."
                            raise RuntimeError(msg)
                _orig(self, event)

            cls.handle_event = _guarded  # type: ignore[method-assign]

    def __init__(self) -> None:
        """Initialize the EventHandler."""
        self.__stored_values: dict[str, Any] = {}
        self._attached_to: _Attachment = _Attachment.NONE
        self._owner_thread: int | None = None
        self._owner_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:  # noqa: D105
        if self._owner_thread is not None:
            msg = "Cannot pickle an event handler after it has been used."
            raise RuntimeError(msg)
        state = self.__dict__.copy()
        state.pop("_owner_lock", None)
        state.pop("_owner_thread", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:  # noqa: D105
        self.__dict__.update(state)
        self._owner_thread = None
        self._owner_lock = threading.Lock()

    def register_dispatcher(self) -> None:
        """Mark this handler as owned by an event dispatcher.

        Raises:
            RuntimeError: If the handler is already registered with a dispatcher
                          or attached to a compute step.
        """
        if self._attached_to is _Attachment.DISPATCHER:
            msg = "This event handler is already registered with a dispatcher."
            raise RuntimeError(msg)
        if self._attached_to is _Attachment.COMPUTE_STEP:
            msg = (
                "This event handler is already registered directly with a compute step."
            )
            raise RuntimeError(msg)
        self._attached_to = _Attachment.DISPATCHER

    def register_compute_step(self) -> None:
        """Mark this handler as owned by one or more compute steps.

        Raises:
            RuntimeError: If the handler is registered with a dispatcher.
        """
        if self._attached_to is _Attachment.DISPATCHER:
            msg = "This event handler is already registered with a dispatcher."
            raise RuntimeError(msg)
        self._attached_to = _Attachment.COMPUTE_STEP

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
        self.__stored_values[key] = value
