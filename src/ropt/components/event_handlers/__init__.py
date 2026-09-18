"""Event handlers: objects that react to the events a compute step emits.

[`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] keeps the best
result, [`HistoryHandler`][ropt.components.event_handlers.HistoryHandler] keeps
every result, [`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler]
builds a table, and [`CallbackHandler`][ropt.components.event_handlers.CallbackHandler]
forwards selected events to a callback.
[`EventForwardHandler`][ropt.components.event_handlers.EventForwardHandler]
forwards them to an
[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher], which
delivers events from the asyncio event loop's thread, so handlers shared across
concurrent compute steps need no locking. See
[Optimization Workflows](../advanced/workflows.md) for usage.
"""

from __future__ import annotations

from ._callback_handler import CallbackHandler
from ._dataframe_handler import DataFrameHandler
from ._event_dispatcher import EventDispatcher
from ._forward_handler import EventForwardHandler
from ._history_handler import HistoryHandler
from ._results_handler import ResultsHandler
from .base import EventHandler

__all__ = [
    "CallbackHandler",
    "DataFrameHandler",
    "EventDispatcher",
    "EventForwardHandler",
    "EventHandler",
    "HistoryHandler",
    "ResultsHandler",
]
