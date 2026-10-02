"""Progress reporting for the high-level API.

A report callback is the small counterpart of a result handler: it is wired up
as an ordinary handler, but as one belonging to a single run, which is why it is
given per run even where `handlers=` only takes shared groups.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ropt.components.event_handlers import CallbackHandler
from ropt.enums import EnOptEventType
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from ropt.components.event_handlers import EventHandler
    from ropt.events import EnOptEvent


ReportCallback = Callable[[FunctionResults], bool | None]
"""Called with each function evaluation; returning `True` stops the run.

Results later in the same batch are not passed on.
"""


def make_report_handler(report: ReportCallback) -> EventHandler:
    def _callback(event: EnOptEvent) -> None:
        for item in event.results or ():
            if (
                isinstance(item, FunctionResults)
                and report(item)
                and event.source is not None
            ):
                # A truthy return asks the emitting run to stop; the break is
                # what makes reporting end there rather than run out the batch.
                event.source.stop()
                break

    return CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_callback
    )
