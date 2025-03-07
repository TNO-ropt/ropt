"""This module implements the default tracker handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

if TYPE_CHECKING:
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
        tags: set[str] | None = None,
    ) -> None:
        """Initialize a default store results handler object.

        The `tags` field allows optional labels to be attached to each result,
        assisting result handlers in filtering relevant results.

        Args:
            plan:                 The plan that runs this step.
            tags:                 Tags to filter the sources to track.
        """
        super().__init__(plan)
        self._tags = set() if tags is None else tags
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle an event.

        Args:
            event: The event to handle.
        """
        if (
            event.event_type == EventType.FINISHED_EVALUATION
            and event.tag in self._tags
        ):
            results = event.data.get("results", None)
            transformed_results = event.data.get("transformed_results", results)
            if transformed_results is not None:
                self["results"] = (
                    transformed_results
                    if self["results"] is None
                    else (*self["results"], *transformed_results)
                )
