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

from ._result_callback import make_result_callback_handler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt._handlers import EventHandler

    from ._result_callback import ResultCallback


def make_handlers(
    handlers: Sequence[EventHandler] | None,
    result_callback: ResultCallback | None,
    *,
    report_gradients: bool = False,
) -> list[EventHandler]:
    ordered = list(handlers or ())
    if result_callback is not None:
        ordered.append(
            make_result_callback_handler(result_callback, gradients=report_gradients)
        )
    return ordered
