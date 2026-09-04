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

from ._result import EvaluateResult, _build_evaluate_result

if TYPE_CHECKING:
    from ropt.components.event_handlers import EventHandler
    from ropt.events import EnOptEvent


ReportCallback = Callable[[EvaluateResult], bool | None]


def make_report_handler(report: ReportCallback) -> EventHandler:
    """Build a handler that reports each new function evaluation.

    The results are unscaled and adapted to an
    `EvaluateResult` before the callback is invoked; gradient results are
    skipped. If the callback returns `True`, the emitting run is asked to stop
    gracefully (exit code `USER_ABORT`); any other return value continues it.
    Reporting stops there: results after it in the same batch are not passed on.

    Args:
        report: The callback invoked with an `EvaluateResult` per evaluation.

    Returns:
        A handler forwarding each function evaluation to the callback.
    """

    def _callback(event: EnOptEvent) -> None:
        for item in event.results or ():
            unscaled = item.unscale(event.context)
            if (
                isinstance(unscaled, FunctionResults)
                and report(_build_evaluate_result(unscaled))
                and event.source is not None
            ):
                # A truthy return asks the emitting run to stop; the break is
                # what makes reporting end there rather than run out the batch.
                event.source.stop()
                break

    return CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_callback
    )
