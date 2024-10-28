r"""Code for executing optimization plans.

The [`Plan`][ropt.plan.Plan] class is responsible for executing optimization
plans, accepting optional input values and returning output values. The plan's
configuration is managed by the [`PlanConfig`][ropt.config.plan.PlanConfig]
class, which specifies the inputs, outputs, variables, steps, and result
handlers.

A plan comprises [`SetStep`][ropt.plan.SetStep] and
[`RunStep`][ropt.plugins.plan.base.RunStep] objects that define individual actions, as well
as [`ResultHandler`][ropt.plugins.plan.base.ResultHandler] objects that process and store
data generated during execution. Both run steps and result handlers are
implemented through a [`plugin`][ropt.plugins.plan] mechanism, making it easy to
extend the range of supported steps and result handlers. The `ropt` library also
includes a [default plan plugin][ropt.plugins.plan.default.DefaultPlanPlugin]
that provides various run steps and result handlers to support a broad range of
optimization workflows.

Most optimization plans require shared state across all steps, such as a
callable for function evaluations or a random number generator. This shared
state is provided through an [`OptimizerContext`][ropt.plan.OptimizerContext]
object, which is supplied when creating the plan.

Setting up and executing a plan object for simple optimization cases can be
complex. The [`OptimizationPlanRunner`][ropt.plan.OptimizationPlanRunner] class
simplifies this process by providing a convenient way to build and execute
straightforward plans involving a single optimization.
"""

from ._events import Event
from ._expr import ExpressionEvaluator
from ._plan import OptimizerContext, Plan
from ._run import OptimizationPlanRunner
from ._set import SetStep

__all__ = [
    "Event",
    "ExpressionEvaluator",
    "OptimizationPlanRunner",
    "OptimizerContext",
    "Plan",
    "SetStep",
]
