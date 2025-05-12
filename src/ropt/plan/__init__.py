r"""Code for executing optimization plans.

The [`Plan`][ropt.plan.Plan] class orchestrates optimization workflows by
managing steps and event handlers.

A plan consists of [`PlanStep`][ropt.plugins.plan.base.PlanStep] objects, which
define individual actions, and
[`EventHandler`][ropt.plugins.plan.base.EventHandler] objects, which process and
store data generated during execution. Both steps and event handlers are
implemented using a [plugin][ropt.plugins.plan] mechanism, making it easy to
extend the range of supported actions and data processing. The `ropt` library
provides default implementations through the [default plan
handler][ropt.plugins.plan.default.DefaultEventHandlerPlugin] and [default plan
step][ropt.plugins.plan.default.DefaultPlanStepPlugin] plugins. These provide
basic steps and event handlers to support a wide range of optimization
workflows.

Setting up and executing a `Plan` object for simple optimization cases can be
complex. The [`BasicOptimizer`][ropt.plan.BasicOptimizer] class simplifies this
process by providing a convenient way to build and execute straightforward plans
involving a single optimization.
"""

from ._basic_optimizer import BasicOptimizer
from ._evaluator import create_evaluator
from ._events import Event
from ._plan import Plan

__all__ = ["BasicOptimizer", "Event", "Plan", "create_evaluator"]
