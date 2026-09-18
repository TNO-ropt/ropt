"""The engines the workflow components are built on.

[`EnsembleEvaluator`][ropt.core.EnsembleEvaluator] orchestrates the
per-realization function calls of one evaluation, and
[`EnsembleOptimizer`][ropt.core.EnsembleOptimizer] drives the configured
backend. Both are exposed for plugin authors and workflow developers; see
[Optimization Workflows](../advanced/workflows.md) for the layer above them.
"""

from ._callback import OptimizerCallback, OptimizerCallbackResult
from ._evaluator import EnsembleEvaluator
from ._optimizer import EnsembleOptimizer, SignalEvaluationCallback

__all__ = [
    "EnsembleEvaluator",
    "EnsembleOptimizer",
    "OptimizerCallback",
    "OptimizerCallbackResult",
    "SignalEvaluationCallback",
]
