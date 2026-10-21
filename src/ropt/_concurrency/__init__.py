"""Concurrency primitives for workflow coordinators.

[`AbortScope`][ropt._concurrency.AbortScope] aborts work
associated with it. For the helpers meant for scripts and applications, see
[`ropt.utils`][ropt.utils].

[`AbortScope.as_current`][ropt._concurrency.AbortScope.as_current]
makes a scope current for the code that a call is running, and
[`current_abort_scope`][ropt._concurrency.current_abort_scope]
returns it. A call made from that code is nested in the current scope.
"""

from __future__ import annotations

from ._abort_scope import AbortScope, aborting, current_abort_scope
from ._program import run_program

__all__ = [
    "AbortScope",
    "aborting",
    "current_abort_scope",
    "run_program",
]
