"""Evaluators: the batch callables used to evaluate the model.

[`make_batch_evaluator`][ropt._evaluators.make_batch_evaluator]
builds one from a function that is called once per row. The batch types it
exchanges are in `ropt._evaluation`. See
[Evaluating in Parallel](../running/parallel.md) for usage.
"""

from __future__ import annotations

from ._counter import BatchIdCounter
from ._evaluator import make_batch_evaluator
from .base import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)

__all__ = [
    "BatchIdCounter",
    "EvaluationFunctionCallback",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "make_batch_evaluator",
]
