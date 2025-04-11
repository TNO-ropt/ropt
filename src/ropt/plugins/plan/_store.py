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
        transforms: OptModelTransforms | None = None,
    ) -> None:
        """Initialize a default store results handler object.

        This handler stores a collection of `Results` objects from specified
        sources. It accumulates results from `FINISHED_EVALUATION` events
        emitted by the designated sources.

        The `sources` parameter determines which steps' results are tracked.

        The stored results can be accessed via the `"results"` key. The
        results are stored as a tuple of `Results` objects.

        If the `transforms` argument is not None, these are applied to transform
        the results from the optimizer domain to the user domain.

        Args:
            plan:       The plan that this handler is part of.
            sources:    The IDs of the steps whose results should be tracked.
            transforms: Optional transforms to apply to the results.
        """
        super().__init__(plan)
        self._sources = set() if sources is None else sources
        self._transforms = transforms
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
            if (results := event.data.get("results", None)) is None:
                return
            if self._transforms is not None:
                results = (
                    item.transform_from_optimizer(self._transforms) for item in results
                )
            self["results"] = (
                results if self["results"] is None else (*self["results"], *results)
            )
