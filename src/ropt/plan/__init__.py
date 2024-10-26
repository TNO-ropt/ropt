r"""Code for executing optimization plans.

The [`Plan`][ropt.plan.Plan] class is responsible for executing optimization
plans, accepting optional input values and returning output values. The plan's
configuration is handled by the [`PlanConfig`][ropt.config.plan.PlanConfig]
class, which specifies the inputs, outputs, variables, steps, and result
handlers.

A plan comprises [`PlanStep`][ropt.plugins.plan.base.PlanStep] objects, which define
individual steps, and [`ResultHandler`][ropt.plugins.plan.base.ResultHandler]
objects, which process and store data generated during execution. Both step and
handler objects are implemented through a [`plugin`][ropt.plugins.plan]
mechanism, making it easy to extend the range of supported steps and result
handlers. The `ropt` library also includes a [default plan
plugin][ropt.plugins.plan.default.DefaultPlanPlugin] that provides a variety of
steps and result handlers to support a broad range of optimization workflows.

Most optimization plans require shared state across all steps, such as a
callable for function evaluations or a random number generator. This shared
state is provided through an [`OptimizerContext`][ropt.plan.OptimizerContext]
object, supplied when creating the plan.

Setting up and executing a plan object for simple optimization cases can be
complex. The [`OptimizationPlanRunner`][ropt.plan.OptimizationPlanRunner] class
simplifies this process, providing a convenient way to build and execute
straightforward plans involving a single optimization.
"""

from ropt.optimization import EnsembleOptimizer

from ._events import Event
from ._plan import OptimizerContext, Plan
from ._run import OptimizationPlanRunner

__all__ = [
    "OptimizationPlanRunner",
    "EnsembleOptimizer",
    "Event",
    "OptimizerContext",
    "Plan",
]
