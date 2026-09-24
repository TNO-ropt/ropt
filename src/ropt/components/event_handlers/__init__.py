"""Event handlers: objects that react to the events a compute step emits.

[`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] keeps the best
result, [`HistoryHandler`][ropt.components.event_handlers.HistoryHandler] keeps
every result, [`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler]
builds a table, and [`CallbackHandler`][ropt.components.event_handlers.CallbackHandler]
forwards selected events to a callback.

A handler serializes its own calls, so the same handler may be attached to
several compute steps running at once. See
[Optimization Workflows](../advanced/workflows.md) for usage.
"""

from __future__ import annotations

from ._callback_handler import CallbackHandler
from ._dataframe_handler import DataFrameHandler
from ._history_handler import HistoryHandler
from ._results_handler import ResultsHandler
from .base import EventHandler

__all__ = [
    "CallbackHandler",
    "DataFrameHandler",
    "EventHandler",
    "HistoryHandler",
    "ResultsHandler",
]
