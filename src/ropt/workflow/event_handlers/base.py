"""Base classes for event handler plugins and event handlers."""

from __future__ import annotations

import functools
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

    Warning:
        Event handler instances must not be called concurrently from multiple
        threads. Do not attach the same handler instance to compute steps that
        run in parallel.
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
                if not self._in_use.acquire(blocking=False):
                    msg = (
                        "EventHandler does not support concurrent use across threads; "
                        "do not attach the same handler to compute steps running in parallel."
                    )
                    raise RuntimeError(msg)
                try:
                    _orig(self, event)
                finally:
                    self._in_use.release()

            cls.handle_event = _guarded  # type: ignore[method-assign]

    def __init__(self) -> None:
        """Initialize the EventHandler."""
        self.__stored_values: dict[str, Any] = {}
        self._in_use = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:  # noqa: D105
        state = self.__dict__.copy()
        state.pop("_in_use", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:  # noqa: D105
        self.__dict__.update(state)
        self._in_use = threading.Lock()

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
