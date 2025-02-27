r"""Code for executing optimization plans.

The [`Plan`][ropt.plan.Plan] class is responsible managing steps and result handlers.

A plan comprises [`PlanStep`][ropt.plugins.plan.base.PlanStep] objects that define
individual actions, as well as
[`ResultHandler`][ropt.plugins.plan.base.ResultHandler] objects that process and
store data generated during execution. Both steps and result handlers are
implemented through a [`plugin`][ropt.plugins.plan] mechanism, making it easy to
extend the range of supported steps and result handlers. The `ropt` library also
includes [default plan handler][ropt.plugins.plan.default.DefaultPlanHandlerPlugin] and
[default plan step][ropt.plugins.plan.default.DefaultPlanStepPlugin]

that provides various steps and result handlers to support a broad range of
optimization workflows.

Most optimization plans require shared state across all steps, such as a
callable for function evaluations or a random number generator. This shared
state is provided through an [`OptimizerContext`][ropt.plan.OptimizerContext]
object, which is supplied when creating the plan.

Setting up and executing a plan object for simple optimization cases can be
complex. The [`BasicOptimizer`][ropt.plan.BasicOptimizer] class simplifies this
process by providing a convenient way to build and execute straightforward plans
involving a single optimization.
"""

from ._basic_optimizer import BasicOptimizer
from ._context import OptimizerContext
from ._events import Event
from ._plan import Plan

__all__ = [
    "BasicOptimizer",
    "Event",
    "OptimizerContext",
    "Plan",
]
