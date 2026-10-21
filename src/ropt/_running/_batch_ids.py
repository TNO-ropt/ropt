"""The batch IDs a run draws from.

One counter for the whole program, so that several runs reaching the same
handler label their batches apart. The IDs are unique within a process: a
counter holds a lock and cannot be sent to a worker process, and neither can an
event handler.
"""

from __future__ import annotations

import threading


class BatchIdCounter:
    """A thread-safe counter for generating sequential batch IDs.

    Provides a simple default `batch_id_callback` for evaluators. Each call
    returns the next integer starting from zero.

    Pass the same instance to multiple evaluators to share a single counter
    across them, so that their batch IDs remain unique.
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


next_batch_id = BatchIdCounter()
