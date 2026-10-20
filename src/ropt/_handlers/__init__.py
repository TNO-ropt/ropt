"""Event handlers: objects that react to the events a compute step emits.

[`ResultsHandler`][ropt.ResultsHandler] keeps the best
result, [`HistoryHandler`][ropt.HistoryHandler] keeps
every result, [`DataFrameHandler`][ropt.DataFrameHandler]
builds a table, and [`CallbackHandler`][ropt._handlers.CallbackHandler]
forwards selected events to a callback.

A handler serializes its own calls, so the same handler may be attached to
several compute steps running at once.
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
