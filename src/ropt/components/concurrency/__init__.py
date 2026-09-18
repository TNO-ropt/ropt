"""Concurrency primitives for workflow coordinators.

[`run_concurrent`][ropt.components.concurrency.run_concurrent] runs blocking
calls on dedicated threads rather than on the event loop's shared thread pool,
which is what lets many optimizations run at once without starving each other.
For the helpers meant for scripts and applications, see
[`ropt.utils`][ropt.utils].
"""

from __future__ import annotations

from ._run_concurrent import run_concurrent

__all__ = [
    "run_concurrent",
]
