"""Concurrency primitives for workflow coordinators.

[`run_concurrent`][ropt.components.concurrency.run_concurrent] runs blocking
calls on dedicated threads rather than on a shared, bounded thread pool, which
is what lets many optimizations run at once without starving each other.
[`StopSignal`][ropt.components.concurrency.StopSignal] is the other half: one
signal stops every compute step that was given it. For the helpers meant for
scripts and applications, see [`ropt.utils`][ropt.utils].
"""

from __future__ import annotations

from ._run_concurrent import run_concurrent
from ._stop_signal import StopSignal

__all__ = [
    "StopSignal",
    "run_concurrent",
]
