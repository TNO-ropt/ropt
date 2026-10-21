"""Build the ordered event handlers of a run.

Every entry point takes `handlers=`, a list of
[`EventHandler`][ropt.EventHandler] objects. The same
handler may be given to several runs at once, sequential or concurrent, because
[`handle_event`][ropt.EventHandler.handle_event]
serializes its own calls.

See [Result Handlers](../results/handlers.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._report import make_report_handler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt._handlers import EventHandler

    from ._report import ReportCallback


def make_handlers(
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    *,
    report_gradients: bool = False,
) -> list[EventHandler]:
    ordered = list(handlers or ())
    if report is not None:
        ordered.append(make_report_handler(report, gradients=report_gradients))
    return ordered
