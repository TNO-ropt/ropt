"""The main functionality for ensemble-based optimizations."""

from ._callback import OptimizerCallback, OptimizerCallbackResult
from ._events import Event
from ._optimizer import (
    EnsembleOptimizer,
    NestedOptimizerCallback,
    SignalEvaluationCallback,
)

__all__ = [
    "EnsembleOptimizer",
    "Event",
    "NestedOptimizerCallback",
    "OptimizerCallback",
    "OptimizerCallbackResult",
    "SignalEvaluationCallback",
]
