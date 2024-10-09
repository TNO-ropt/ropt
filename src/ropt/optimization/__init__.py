"""The optimizer and event classes.

The [`EnsembleOptimizer`][ropt.optimization.EnsembleOptimizer] class runs
ensemble based optimizations. It is not recommended to used them directly to run
optimizations, the [`Plan`][ropt.plan.Plan] functionality is usually more
flexible and easier to use.
"""

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
    "SignalEvaluationCallback",
]
