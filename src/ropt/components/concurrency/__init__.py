"""Concurrency primitives for workflow coordinators.

[`run_concurrent`][ropt.components.concurrency.run_concurrent] runs blocking
calls on dedicated threads rather than on a shared, bounded thread pool, which
is what lets many optimizations run at once without starving each other.
[`AbortSignal`][ropt.components.concurrency.AbortSignal] is the other half: one
signal cuts off every compute step that was given it. For the helpers meant for
scripts and applications, see [`ropt.utils`][ropt.utils].
"""

from __future__ import annotations

from ._abort_signal import AbortSignal
from ._run_concurrent import run_concurrent

__all__ = [
    "AbortSignal",
    "run_concurrent",
]
