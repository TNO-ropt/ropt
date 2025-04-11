"""This module implements the default tracker handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import EventType
from ropt.plugins.plan.base import PlanHandler

if TYPE_CHECKING:
    import uuid

    from ropt.plan import Event, Plan
    from ropt.transforms import OptModelTransforms


class DefaultStoreHandler(PlanHandler):
    """The default handler for storing optimization results.

    This handler listens for
    [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
    emitted by specified `sources` (plan steps). It collects all
    [`Results`][ropt.results.Results] objects contained within these events and
    stores them sequentially in memory.

    The `sources` parameter filters which steps' results are collected. If a
    `transforms` object is provided, the results are converted from the
    optimizer domain to the user domain before being stored.

    The accumulated results are stored as a tuple and can be accessed via
    dictionary access using the key `"results"` (e.g., `handler["results"]`).
    Each time new results are received from a valid source, they are appended to
    this tuple.
    """

    def __init__(
        self,
        plan: Plan,
        *,
        sources: set[uuid.UUID] | None = None,
        transforms: OptModelTransforms | None = None,
    ) -> None:
        """Initialize a default store results handler.

        This handler collects and stores all [`Results`][ropt.results.Results]
        objects received from specified `sources` (plan steps). It listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        and appends the results contained within them to an internal tuple.

        The `sources` parameter acts as a filter, determining which plan steps
        this handler should listen to. It should be a set containing the unique
        IDs (UUIDs) of the `PlanStep` instances whose results you want to store.
        When a [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION]
        event occurs, this handler checks if the ID of the step that emitted the
        event (`event.source`) is present in the `sources` set. If it is, the
        handler stores the results; otherwise, it ignores the event. If
        `sources` is `None` (the default), the handler will not store results
        from any source.

        If a `transforms` object is provided, the results are converted from the
        optimizer domain to the user domain *before* being stored.

        The accumulated results are stored as a tuple and can be accessed via
        dictionary access using the key `"results"` (e.g., `handler["results"]`).
        Initially, `handler["results"]` is `None`.

        Args:
            plan:       The parent plan instance.
            sources:    Optional set of step UUIDs whose results should be stored.
            transforms: Optional transforms object to apply before storing results.
        """
        super().__init__(plan)
        self._sources = set() if sources is None else sources
        self._transforms = transforms
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by the parent plan. It specifically
        listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        originating from steps whose IDs are included in the `sources` set
        configured during initialization.

        If a relevant event containing results is received, this method retrieves
        the results, optionally transforms them to the user domain if
        `transforms` were provided, and appends them to the tuple stored in
        `self["results"]`.

        Args:
            event: The event object emitted by the plan.
        """
        if (
            event.event_type == EventType.FINISHED_EVALUATION
            and event.source in self._sources
        ):
            if (results := event.data.get("results", None)) is None:
                return
            if self._transforms is not None:
                results = (
                    item.transform_from_optimizer(self._transforms) for item in results
                )
            self["results"] = (
                results if self["results"] is None else (*self["results"], *results)
            )
