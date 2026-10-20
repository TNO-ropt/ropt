"""Concurrency primitives for workflow coordinators.

[`AbortScope`][ropt.components.concurrency.AbortScope] aborts every compute
step that was given it. For the helpers meant for scripts and applications, see
[`ropt.utils`][ropt.utils].

[`AbortScope.as_current`][ropt.components.concurrency.AbortScope.as_current]
makes a scope current for the code that a call is running, and
[`current_abort_scope`][ropt.components.concurrency.current_abort_scope]
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
