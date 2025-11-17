"""This module implements the default store event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EventType
from ropt.plugins.plan.base import EventHandler

if TYPE_CHECKING:
    from ropt.optimization import Event


class DefaultStoreHandler(EventHandler):
    """The default event handler for storing optimization results.

    This event handler listens for
    [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
    emitted by specified steps. It collects all
    [`Results`][ropt.results.Results] objects contained within these events and
    stores them sequentially in memory.

    The accumulated results are stored as a tuple and can be accessed via
    dictionary access using the key `"results"` (e.g., `handler["results"]`).
    Each time new results are received from a valid source, they are appended to
    this tuple.
    """

    def __init__(self) -> None:
        """Initialize a default store event handler.

        This event handler collects and stores all
        [`Results`][ropt.results.Results] objects received from specified steps.
        It listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        and appends the results contained within them to an internal tuple.

        The results are converted from the optimizer domain to the user domain
        *before* being stored. The accumulated results are stored as a tuple and
        can be accessed via dictionary access using the key `"results"` (e.g.,
        `handler["results"]`). Initially, `handler["results"]` is `None`.

        Args:
            plan:       The parent plan instance.
        """
        super().__init__()
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by the parent plan. It specifically
        listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        originating from steps that are connected to the handler.

        If a relevant event containing results is received, this method
        retrieves the results, optionally transforms them to the user domain and
        appends them to the tuple stored in `self["results"]`.

        Args:
            event: The event object emitted by the plan.
        """
        if (results := event.data.get("results")) is None:
            return
        transforms = event.data["transforms"]
        results = (
            item
            if transforms is None
            else item.transform_from_optimizer(event.data["config"], transforms)
            for item in results
        )
        self["results"] = tuple(
            results if self["results"] is None else (*self["results"], *results)
        )

    @property
    def event_types(self) -> set[EventType]:
        """Return the event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EventType.FINISHED_EVALUATION}
