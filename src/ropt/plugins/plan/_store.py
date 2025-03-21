"""This module implements the default tracker handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

if TYPE_CHECKING:
    import uuid

    from ropt.plan import Event, Plan


class DefaultStoreHandler(ResultHandler):
    """The default store results handler object.

    This handler tracks the
    [`Results`][ropt.results.Results] objects that it receives and stores them
    in memory.
    """

    def __init__(
        self,
        plan: Plan,
        *,
        sources: set[uuid.UUID] | None = None,
    ) -> None:
        """Initialize a default store results handler object.

        This handler stores a collection of `Results` objects from specified
        sources. It accumulates results from `FINISHED_EVALUATION` events
        emitted by the designated sources.

        The `sources` parameter determines which steps' results are tracked.
        Only results from steps whose IDs are included in this set will be
        stored. If `sources` is not provided or is `None`, results from all
        sources are tracked.

        The stored results can be accessed via the `"results"` key. The
        results are stored as a tuple of `Results` objects.

        Args:
            plan:    The plan that this handler is part of.
            sources: The IDs of the steps whose results should be tracked.
        """
        super().__init__(plan)
        self._sources = set() if sources is None else sources
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle an event.

        Args:
            event: The event to handle.
        """
        if (
            event.event_type == EventType.FINISHED_EVALUATION
            and event.source in self._sources
        ):
            results = event.data.get("results", None)
            transformed_results = event.data.get("transformed_results", results)
            if transformed_results is not None:
                self["results"] = (
                    transformed_results
                    if self["results"] is None
                    else (*self["results"], *transformed_results)
                )
