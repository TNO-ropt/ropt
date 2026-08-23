r"""Optimization workflow building blocks.

The building blocks live in submodules: `compute_steps`, `event_handlers`,
`evaluators`, `executors`, and `concurrency`.

They fit together in one direction. A **compute step** drives a run and emits
events about it; **event handlers** receive those events; an **evaluator**
produces the function values the step asks for; and an **executor** is where an
evaluator sends the work, if it sends it anywhere. Each is given to the next
explicitly, so a workflow is whatever the caller wires up: nothing here looks
around for the parts it needs.

None of these objects can be used across a process boundary. They may be pickled
as part of a payload, but arrive in a worker as inert placeholders, which the
worker reports rather than silently using.
"""
