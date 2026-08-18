"""Building blocks for authoring custom runs on the high-level session.

The names here let a custom compute step or a concurrent launcher plug into the
same background session, executor, and shared handlers that
`ropt.simple.optimize` and `ropt.simple.optimize_many` use. They are kept out of
the `ropt.simple` namespace on purpose: end users do not need them, and importing
from `ropt.simple.compose` marks the code as advanced use.
"""

from __future__ import annotations

from ._evaluator import run_step
from ._handlers import HandlerScope, current_handlers
from ._session import current_executor, gather_shared

__all__ = [
    "HandlerScope",
    "current_executor",
    "current_handlers",
    "gather_shared",
    "run_step",
]
