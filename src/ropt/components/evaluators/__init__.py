"""Evaluators: the objects a compute step uses to evaluate the model.

Each is an [`Evaluator`][ropt.components.evaluators.Evaluator] subclass. For the
plain callable protocols they wrap, see
[Evaluation Classes](../reference/evaluation.md). See
[Writing Evaluation Callbacks](../advanced/evaluation_callbacks.md) and
[Parallel Evaluation](../advanced/parallel.md) for usage.
"""

from __future__ import annotations

from ._batch_evaluator import BatchEvaluator
from ._counter import BatchIdCounter
from ._function_evaluator import FunctionEvaluator
from ._parallel_evaluator import ParallelEvaluator
from .base import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    Evaluator,
)

__all__ = [
    "BatchEvaluator",
    "BatchIdCounter",
    "EvaluationFunctionCallback",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "Evaluator",
    "FunctionEvaluator",
    "ParallelEvaluator",
]
