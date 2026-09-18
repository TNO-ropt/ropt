"""This module implements the result-history event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EnOptEventType

from .base import EventHandler

if TYPE_CHECKING:
    from ropt.events import EnOptEvent
    from ropt.results import Results


class HistoryHandler(EventHandler):
    """Collect all optimization results into a tuple.

    Listens for `FINISHED_EVALUATION` events and appends every
    [`Results`][ropt.results.Results] object to a growing tuple accessible
    via the [`results`][ropt.components.event_handlers.HistoryHandler.results]
    property or `handler["results"]`.

    See [Result Handlers](../running/handlers.md#historyhandler) for full
    details on scaling and accumulation behavior.
    """

    def __init__(self) -> None:
        """Initialize the HistoryHandler."""
        super().__init__()
        self["results"] = None

    @property
    def results(self) -> tuple[Results, ...]:
        """All results collected so far, in the order received."""
        collected: tuple[Results, ...] | None = self["results"]
        return () if collected is None else collected

    def _handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Processes `FINISHED_EVALUATION` events by appending their results to
        `self["results"]`.

        Args:
            event: The event object.
        """
        if event.results:
            self["results"] = tuple(
                event.results
                if self["results"] is None
                else (*self["results"], *event.results)
            )

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EnOptEventType.FINISHED_EVALUATION}
