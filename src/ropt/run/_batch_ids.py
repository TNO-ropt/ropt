"""The batch IDs a run draws from.

One counter for the whole program, so that several runs reaching the same
handler label their batches apart. The IDs are unique within a process: a
counter holds a lock and cannot be sent to a worker process, and neither can an
event handler.
"""

from __future__ import annotations

from ropt.components.evaluators import BatchIdCounter

next_batch_id = BatchIdCounter()
