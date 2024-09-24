"""The optimizer and event classes.

The [`EnsembleOptimizer`][ropt.optimization.EnsembleOptimizer] class runs
ensemble based optimizations. The [`EventBroker`][ropt.optimization.EventBroker]
class is used to handle events emitted during an optimization. It is not
recommended to used them directly to run optimizations, the
[`Plan`][ropt.plan.Plan] functionality is usually more flexible and easier to use.
"""

from ._events import Event, EventBroker
from ._optimizer import (
    EnsembleOptimizer,
    NestedOptimizerCallback,
    SignalEvaluationCallback,
)

__all__ = [
    "EnsembleOptimizer",
    "Event",
    "EventBroker",
    "NestedOptimizerCallback",
    "SignalEvaluationCallback",
]
