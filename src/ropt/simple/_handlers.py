"""Wiring a run's result handlers to its compute step.

Every entry point takes `handlers=`, a list of
[`EventHandler`][ropt.components.event_handlers.EventHandler] objects. The same
handler may be given to several runs at once, sequential or concurrent, because
[`handle_event`][ropt.components.event_handlers.EventHandler.handle_event]
serializes its own calls.

See [Result Handlers](../running/handlers.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._report import make_report_handler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.components.compute_steps import ComputeStep
    from ropt.components.event_handlers import EventHandler

    from ._report import ReportCallback


def attach_handlers(
    step: ComputeStep[Any],
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
) -> None:
    """Wire a run's handlers to its compute step.

    Args:
        step:     The compute step of the run.
        handlers: The handlers to wire up, in the order they are called in.
        report:   An optional callback wired up as a report handler.
    """
    for handler in handlers or ():
        step.add_event_handler(handler)
    if report is not None:
        step.add_event_handler(make_report_handler(report))
