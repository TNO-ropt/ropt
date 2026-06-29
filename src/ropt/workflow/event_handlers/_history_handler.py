"""This module implements the default store event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EnOptEventType

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Generator

    from ropt.events import EnOptEvent
    from ropt.results import DomainType, Results


class HistoryHandler(EventHandler):
    """Collect all optimization results into a tuple.

    Listens for `FINISHED_EVALUATION` events and appends every
    [`Results`][ropt.results.Results] object to a growing tuple accessible
    via `handler["results"]`.

    See [Optimization Workflows](../usage/workflows.md#history) for full
    details on domain handling and accumulation behavior.

    Thread safety:
        `handle_event` is serialized by an internal lock, so the same
        instance may be attached to compute steps that run concurrently in
        different threads. The stored value (`handler["results"]`) is always
        an immutable tuple (or `None`); callers must not mutate it. When the
        handler is shared across concurrent steps the relative order of
        results from different steps is non-deterministic, but no result is
        lost.
    """

    def __init__(self, *, domain: DomainType = "user") -> None:
        """Initialize the HistoryHandler.

        Args:
            domain: Domain in which to store results ('user' or 'optimizer').
        """
        super().__init__()
        self["results"] = None
        self._domain = domain

    def handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Processes `FINISHED_EVALUATION` events, optionally transforms results
        to the user domain, and appends them to `self["results"]`.

        Args:
            event: The event object.
        """
        with self.locked():
            results: tuple[Results, ...] | Generator[Results, None, None]
            if results := event.results:
                if self._domain == "user":
                    results = (
                        item.transform_from_optimizer(event.context) for item in results
                    )
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
