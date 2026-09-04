"""This module implements the default store event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EnOptEventType

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Generator

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

    def __init__(self, *, scaled: bool = False) -> None:
        """Initialize the HistoryHandler.

        Args:
            scaled: If `True`, store the values as the optimizer works with
                them: scaled and offset, with objectives and gradients negated
                where `maximize` is set. By default the values are unscaled
                first, restoring the quantities as configured.
        """
        super().__init__()
        self["results"] = None
        self._scaled = scaled

    @property
    def results(self) -> tuple[Results, ...]:
        """All results collected so far, in the order received."""
        collected: tuple[Results, ...] | None = self["results"]
        return () if collected is None else collected

    def _handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Processes `FINISHED_EVALUATION` events, unscales the results unless
        scaled values were requested, and appends them to `self["results"]`.

        Args:
            event: The event object.
        """
        results: tuple[Results, ...] | Generator[Results, None, None]
        results = event.results
        if results:
            if not self._scaled:
                results = (item.unscale(event.context) for item in results)
            self["results"] = tuple(
                results if self["results"] is None else (*self["results"], *results)
            )

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EnOptEventType.FINISHED_EVALUATION}
