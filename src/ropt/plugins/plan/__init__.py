"""Framework and Implementations for Optimization Plan Plugins.

This module provides the core components and default implementations for
extending `ropt`'s optimization plan capabilities ([`Plan`][ropt.plan.Plan])
through plugins. It allows users to define custom sequences of operations
(steps) and ways to process the results and events generated during plan
execution (handlers).

**Core Concepts:**

*   **Plan Steps:** Represent individual actions within an optimization plan,
    such as running an optimizer or performing evaluations.
*   **Plan Handlers:** Process events emitted by the plan or its steps, enabling
    tasks like result tracking, data storage, or logging.

The implementation of these core concepts relies on classes derived from the
following abstract base classes:

1.  **Plugin Base Classes:**
    * [`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin]: The base for
      plugins that *create* plan steps. These plugins are discovered by the
      [`PluginManager`][ropt.plugins.PluginManager] and used to instantiate
      actual `PlanStep` objects.
    * [`PlanHandlerPlugin`][ropt.plugins.plan.base.PlanHandlerPlugin]: The base
      for plugins that *create* plan result handlers. Similar to step plugins,
      these are used by the `PluginManager` to instantiate `PlanHandler`
      objects.

2.  **Component Base Classes:**
    * [`PlanStep`][ropt.plugins.plan.base.PlanStep]: The abstract base class
      that all concrete plan step implementations must inherit from. It defines
      the [`run`][ropt.plugins.plan.base.PlanStep.run] method where the step's
      logic resides.
    * [`PlanHandler`][ropt.plugins.plan.base.PlanHandler]: The abstract base
      class for all result handlers. It defines the
      [`handle_event`][ropt.plugins.plan.base.PlanHandler.handle_event] method
      for processing events emitted during plan execution and allows storing
      state using dictionary-like access.

By inheriting from these classes, developers can create custom steps and
handlers that integrate seamlessly into the `ropt` optimization plan execution
framework ([`Plan`][ropt.plan.Plan]).


**Built-in Plan Plugins:**

`ropt` includes default plugins providing common plan components:

*   **Steps** (via
    [`DefaultPlanStepPlugin`][ropt.plugins.plan.default.DefaultPlanStepPlugin]):
    *   `evaluator`: Performs ensemble evaluations
        ([`DefaultEvaluatorStep`][ropt.plugins.plan.evaluator.DefaultEvaluatorStep]).
    *   `optimizer`: Runs an optimization algorithm using a configured optimizer
        plugin
        ([`DefaultOptimizerStep`][ropt.plugins.plan.optimizer.DefaultOptimizerStep]).
*   **Handlers** (via
    [`DefaultPlanHandlerPlugin`][ropt.plugins.plan.default.DefaultPlanHandlerPlugin]):
    *   `tracker`: Tracks the 'best' or 'last' valid result based on objective
        value and constraints
        ([`DefaultTrackerHandler`][ropt.plugins.plan._tracker.DefaultTrackerHandler]).
    *   `store`: Accumulates all results from specified sources
        ([`DefaultStoreHandler`][ropt.plugins.plan._store.DefaultStoreHandler]).

These built-in components allow for the construction of standard optimization
workflows out-of-the-box, while the plugin architecture enables customization
and extension.
"""

from .base import PlanHandler, PlanHandlerPlugin, PlanStep, PlanStepPlugin

__all__ = [
    "PlanHandler",
    "PlanHandlerPlugin",
    "PlanStep",
    "PlanStepPlugin",
]
