"""Run an optimization or evaluation and emit its events."""

from __future__ import annotations

from ._evaluator import run_evaluation
from ._optimizer import run_optimization

__all__ = ["run_evaluation", "run_optimization"]
