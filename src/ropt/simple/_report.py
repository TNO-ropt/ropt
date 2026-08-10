"""Progress reporting for the high-level API."""

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

    The results are transformed to the user domain and adapted to an
    `EvaluateResult` before the callback is invoked; gradient results are
    skipped. If the callback returns `True`, the emitting run is asked to stop
    gracefully (exit code `USER_ABORT`); any other return value continues it.

    Args:
        report: The callback invoked with an `EvaluateResult` per evaluation.

    Returns:
        A handler forwarding each function evaluation to the callback.
    """

    def _callback(event: EnOptEvent) -> None:
        for item in event.results or ():
            transformed = item.transform_from_optimizer(event.context)
            if (
                isinstance(transformed, FunctionResults)
                and report(_build_evaluate_result(transformed))
                and event.source is not None
            ):
                event.source.stop()
                break

    return CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_callback
    )
