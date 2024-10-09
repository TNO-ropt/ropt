r"""Code to run optimization plans.

Optimization plans consist of a series of steps that define an optimization
workflow. This can be as simple as a single optimization run or involve
multiple, potentially nested optimizations.

These plans are configured using [PlanConfig][ropt.config.plan.PlanConfig]
objects and executed by a [Plan][ropt.plan.Plan] object. A plan is composed of
[EventHandler][ropt.plugins.plan.base.EventHandler] objects, which process and
store data generated during execution, and
[PlanStep][ropt.plugins.plan.base.PlanStep] objects that define the individual
steps executed by the plan. Both handler and step objects are implemented
through a [plugin][ropt.plugins.plan] mechanism. The ropt library also offers
several [default][ropt.plugins.plan.default.DefaultPlanPlugin] plan objects to
support
various optimization workflows.

A plan can store data identified by a name, known as plan variables. These
variables can be accessed using the `[]` operator, and their existence can be
checked with the `in` operator. Event handler and step objects typically store
values in variables with their names set in their configuration.

Event handler and step objects are configured using their corresponding
configuration objects,
[`EventHandlerConfig`][ropt.config.plan.EventHandlerConfig] and
[`StepConfig`][ropt.config.plan.StepConfig], respectively. These are Pydantic
classes initialized via dictionaries of configuration values. These can be
strings that are interpolated with the variables stored in the plan. Plan
variables, prefixed with the `$` sign, will be substituted with their
corresponding values. Any part of a string enclosed by `${{` and `}}` will be
parsed as a mathematical expression, with variables replaced by their
corresponding values in the plan.

To execute optimization plans, additional shared state may be required across
all steps, such as a callable for function evaluations or a random number
generator. For this purpose, an [OptimizerContext][ropt.plan.OptimizerContext]
object is supplied when creating the plan, which maintains this shared state.

Initializing and executing a plan object for simple optimization cases can be
cumbersome. The [`OptimizationPlanRunner`][ropt.plan.OptimizationPlanRunner]
provides a convenient way to build and execute such plans with ease.
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
