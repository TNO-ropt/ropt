"""This module implements the default meta data handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

from ._utils import _get_set

if TYPE_CHECKING:
    from ropt.plan import Event, Plan


class DefaultMetadataHandler(ResultHandler):
    """The default metadata results handler.

    This handler adds arbitrary metadata to results produced by steps by merging
    them into the `metadata` field of the [`Results`][ropt.results.Results]
    objects that it receives. It is configured using a dictionary that maps keys
    to the data to store. These data entries can be of any type; however, if
    they the values is string starting with a single `$` it is assumed to be the
    key to value stored in the plan that should be inserted into the metadata.
    """

    def __init__(
        self, plan: Plan, *, data: dict[str, Any], tags: str | set[str] | None = None
    ) -> None:
        """Initialize a default metadata results handler.

        The data to merge into the metadata of a result are required.

        The `tags` field allows optional labels to be attached to each result,
        which can assist result handlers in filtering relevant results.

        Args:
            plan: The plan that runs this step.
            data: Data to merge into the metadata of the results.
            tags: Optional tags specifying which result sources to modify.
        """
        super().__init__(plan)

        self._data = data
        self._tags = _get_set(tags)

    def handle_event(self, event: Event) -> Event:
        """Handle an event.

        Args:
            event: The event to handle.

        Returns:
            The (possibly modified) event.
        """
        if (
            event.event_type
            in {
                EventType.FINISHED_EVALUATION,
                EventType.FINISHED_EVALUATOR_STEP,
            }
            and "results" in event.data
            and (event.tags & self._tags)
        ):
            for results in event.data["results"]:
                for key, value in self._data.items():
                    results.metadata[key] = (
                        self.plan[value[1:]]
                        if (
                            isinstance(value, str)
                            and value.startswith("$")
                            and not value[1:].startswith("$")
                        )
                        else value
                    )
        return event
