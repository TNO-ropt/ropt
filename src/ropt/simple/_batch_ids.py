"""The batch IDs a high-level run draws from.

A batch ID labels a batch of evaluations. It distinguishes one batch's results
from another's in a table, and decides nothing about what is computed or where
it runs, which is why one counter for the whole program is enough, and why it
may be ambient state: no run behaves differently for the ID it is handed.

Distinct IDs matter where several runs' results reach the same handler. That is
a different question from where the evaluations run, so the counter is kept
apart from whatever carries the workers.

The IDs are unique within a process. A counter holds a lock and cannot be sent
to a worker process, and neither can an event handler, so results produced in
separate processes never meet in one table.

The low-level API is unaffected: an evaluator built directly, with no
`batch_id_callback`, still gets a private counter of its own.
"""

from __future__ import annotations

from ropt.components.evaluators import BatchIdCounter

next_batch_id = BatchIdCounter()
