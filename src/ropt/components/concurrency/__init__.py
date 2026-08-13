"""Loop-independent concurrency primitives for coordinators."""

from __future__ import annotations

from ._run_concurrent import run_concurrent

__all__ = [
    "run_concurrent",
]
