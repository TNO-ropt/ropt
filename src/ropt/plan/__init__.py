r"""Code for executing optimization plans.

The [`Plan`][ropt.plan.Plan] class orchestrates optimization workflows by
managing steps and result handlers.

A plan consists of [`PlanStep`][ropt.plugins.plan.base.PlanStep] objects, which
define individual actions, and
[`PlanHandler`][ropt.plugins.plan.base.PlanHandler] objects, which process
and store data generated during execution. Both steps and result handlers are
implemented using a [plugin][ropt.plugins.plan] mechanism, making it easy to
extend the range of supported actions and data processing. The `ropt` library
provides default implementations through the [default plan
handler][ropt.plugins.plan.default.DefaultPlanHandlerPlugin] and [default plan
step][ropt.plugins.plan.default.DefaultPlanStepPlugin] plugins. These provide
basic steps and result handlers to support a wide range of optimization
workflows.

Most optimization plans require shared state across all steps, such as the
plugin manager and an evaluator callable for function evaluations. This shared
state is managed by the [`OptimizerContext`][ropt.plan.OptimizerContext] object,
which is provided when creating a plan. The `OptimizerContext` also handles
[events][ropt.plan.Event] produced by plan steps by calling registered callbacks
and forwarding them to result handlers.

Setting up and executing a `Plan` object for simple optimization cases can be
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
