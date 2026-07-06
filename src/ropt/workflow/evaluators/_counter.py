"""Provides the default batch ID counter for evaluators."""

from __future__ import annotations

import threading
from typing import Any


class BatchIdCounter:
    """A thread-safe counter for generating sequential batch IDs.

    Provides a simple default `batch_id_callback` for evaluators. Each call
    returns the next integer starting from zero.

    Pass the same instance to multiple evaluators to share a single counter
    across them — useful in nested or parallel optimization setups where all
    evaluators should produce globally unique batch IDs.

    See [Writing Evaluation Callbacks](../usage/evaluation_callbacks.md) for
    usage details and examples.
    """

    def __init__(self) -> None:
        """Initialize the counter starting at zero."""
        self._value = 0
        self._lock = threading.Lock()

    def __call__(self) -> int:
        """Return the next batch ID and advance the counter."""
        with self._lock:
            value = self._value
            self._value += 1
            return value

    def __getstate__(self) -> dict[str, Any]:
        # threading.Lock is not picklable; drop it and recreate in __setstate__.
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
