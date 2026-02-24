"""This module provides the default plugin implementations for event handlers.

**Supported handlers:**

  - `tracker`: Tracks the 'best' or 'last' valid result based on objective
      value and constraints
      ([`DefaultTrackerHandler`][ropt.plugins.event_handler._tracker.DefaultTrackerHandler]).
  - `store`: Accumulates all results from specified sources
      ([`DefaultStoreHandler`][ropt.plugins.event_handler._store.DefaultStoreHandler]).
  - `observer`: Listens for events from specified sources, and calls a
      callback for each event
      ([`DefaultObserverHandler`][ropt.plugins.event_handler._observer.DefaultObserverHandler]).
  - `table`: Collect results in a set of pandas data frames
      ([`DefaultTableHandler`][ropt.plugins.event_handler._table.DefaultTableHandler]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from ._observer import DefaultObserverHandler
from ._store import DefaultStoreHandler
from ._table import DefaultTableHandler
from ._tracker import DefaultTrackerHandler
from .base import EventHandlerPlugin

if TYPE_CHECKING:
    from .base import EventHandler


_EVENT_HANDLER_OBJECTS: Final[dict[str, type[EventHandler]]] = {
    "observer": DefaultObserverHandler,
    "tracker": DefaultTrackerHandler,
    "store": DefaultStoreHandler,
    "table": DefaultTableHandler,
}


class DefaultEventHandlerPlugin(EventHandlerPlugin):
    """The default plugin for creating built-in event handlers.

    This plugin acts as a factory for the standard `EventHandler`
    implementations provided by `ropt`.

    **Supported Handlers:**

    - `tracker`: Creates a
        [`DefaultTrackerHandler`][ropt.plugins.event_handler._tracker.DefaultTrackerHandler]
        instance, which tracks either the 'best' or 'last' valid result based on
        objective value and constraints.
    - `store`: Creates a
        [`DefaultStoreHandler`][ropt.plugins.event_handler._store.DefaultStoreHandler]
        instance, which accumulates all results received from specified sources.
    - `observer`: Creates a
        [`DefaultObserverHandler`][ropt.plugins.event_handler._observer.DefaultObserverHandler]
        instance, which calls a callback for each event received from specified
        sources.
    """

    @classmethod
    def create(cls, name: str, **kwargs: dict[str, Any]) -> EventHandler:
        """Create an event handler.

        # noqa
        """  # noqa: DOC201, DOC501
        _, _, name = name.lower().rpartition("/")
        obj = _EVENT_HANDLER_OBJECTS.get(name)
        if obj is not None:
            return obj(**kwargs)

        msg = f"Unknown event handler object type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in _EVENT_HANDLER_OBJECTS
