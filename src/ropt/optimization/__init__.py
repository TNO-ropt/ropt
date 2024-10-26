"""The main functionality for ensemble-based optimizations.

The [`EnsembleOptimizer`][ropt.optimization.EnsembleOptimizer] class provides
the core functionality for running ensemble-based optimizations. Direct use of
this class is generally discouraged; instead, the [`Plan`][ropt.plan.Plan] or
[`OptimizationPlanRunner`][ropt.plan.OptimizationPlanRunner] classes are
recommended for more flexibility and ease of use.
"""

from ._optimizer import (
    EnsembleOptimizer,
    NestedOptimizerCallback,
    SignalEvaluationCallback,
)

__all__ = [
    "EnsembleOptimizer",
    "NestedOptimizerCallback",
    "SignalEvaluationCallback",
]
