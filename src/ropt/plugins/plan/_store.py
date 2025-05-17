"""This module implements the default store event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EventType
from ropt.plugins.plan.base import EventHandler, PlanComponent

if TYPE_CHECKING:
    from ropt.plan import Event, Plan


class DefaultStoreHandler(EventHandler):
    """The default event handler for storing optimization results.

    This event handler listens for
    [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
    emitted by specified `sources` (plan steps). It collects all
    [`Results`][ropt.results.Results] objects contained within these events and
    stores them sequentially in memory.

    The `sources` parameter filters which steps' results are collected. The
    accumulated results are stored as a tuple and can be accessed via dictionary
    access using the key `"results"` (e.g., `handler["results"]`). Each time new
    results are received from a valid source, they are appended to this tuple.
    """

    def __init__(
        self,
        plan: Plan,
        tags: set[str] | None = None,
        sources: set[PlanComponent | str] | None = None,
    ) -> None:
        """Initialize a default store event handler.

        This event handler collects and stores all
        [`Results`][ropt.results.Results] objects received from specified
        `sources` (plan steps). It listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        and appends the results contained within them to an internal tuple.

        The `sources` parameter acts as a filter, determining which plan steps
        or tags this event handler should listen to.

        The results are converted from the optimizer domain to the user domain
        *before* being stored. The accumulated results are stored as a tuple and
        can be accessed via dictionary access using the key `"results"` (e.g.,
        `handler["results"]`). Initially, `handler["results"]` is `None`.

        Args:
            plan:       The parent plan instance.
            tags:       Optional tags
            sources:    Optional set of steps whose results should be stored.
        """
        super().__init__(plan, tags, sources)
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by the parent plan. It specifically
        listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        originating from steps that are included in the `sources` set configured
        during initialization.

        If a relevant event containing results is received, this method
        retrieves the results, optionally transforms them to the user domain and
        appends them to the tuple stored in `self["results"]`.

        Args:
            event: The event object emitted by the plan.
        """
        if event.event_type == EventType.FINISHED_EVALUATION:
            if (results := event.data.get("results", None)) is None:
                return
            if event.config.transforms is not None:
                results = (
                    item.transform_from_optimizer(event.config.transforms)
                    for item in results
                )
            self["results"] = (
                results if self["results"] is None else (*self["results"], *results)
            )
