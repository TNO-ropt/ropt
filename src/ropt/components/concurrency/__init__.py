"""Concurrency primitives for workflow coordinators.

[`AbortSignal`][ropt.components.concurrency.AbortSignal] aborts every compute
step that was given it. For the helpers meant for scripts and applications, see
[`ropt.utils`][ropt.utils].

[`AbortSignal.as_parent`][ropt.components.concurrency.AbortSignal.as_parent]
marks the code that a run or offload is running, and
[`parent_signal`][ropt.components.concurrency.parent_signal] returns the signal
it was marked with: the parent signal of a run or offload started from that
code.
"""

from __future__ import annotations

from ._abort_signal import AbortSignal, parent_signal

__all__ = [
    "AbortSignal",
    "parent_signal",
]
