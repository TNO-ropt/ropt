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
*   **Evaluators:** Perform the actual function evaluations (e.g., objective
    functions, constraints) required by the optimization process.

The implementation of these core concepts relies on classes derived from the
following abstract base classes:

1.  **Plugin Base Classes:**
    * [`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin]: The base for
      plugins that *create* plan steps. These plugins are discovered by the
      [`PluginManager`][ropt.plugins.PluginManager] and used to instantiate
      actual `PlanStep` objects.
    * [`EventHandlerPlugin`][ropt.plugins.plan.base.EventHandlerPlugin]: The
      base for plugins that *create* event handlers. Similar to step plugins,
      these are used by the `PluginManager` to instantiate `EventHandler`
      objects.
    * [`EvaluatorPlugin`][ropt.plugins.plan.base.EvaluatorPlugin]: The base for
      plugins that *create* evaluators. These are used by the
      `PluginManager` to instantiate `Evaluator` objects, which are
      responsible for function computations.

2.  **Component Base Classes:**
    * [`PlanStep`][ropt.plugins.plan.base.PlanStep]: The abstract base class
      that all concrete plan step implementations must inherit from. It defines
      the [`run`][ropt.plugins.plan.base.PlanStep.run] method where the step's
      logic resides.
    * [`EventHandler`][ropt.plugins.plan.base.EventHandler]: The abstract base
      class for all event handlers. It defines the
      [`handle_event`][ropt.plugins.plan.base.EventHandler.handle_event] method
      for processing events emitted during plan execution and allows storing
      state using dictionary-like access.
    * [`Evaluator`][ropt.plugins.plan.base.Evaluator]: The abstract base class
      for all evaluators. It defines the
      [`eval`][ropt.plugins.plan.base.Evaluator.eval] method responsible for
      performing function computations.

By inheriting from these classes, developers can create custom steps and
handlers that integrate seamlessly into the `ropt` optimization plan execution
framework ([`Plan`][ropt.plan.Plan]).


**Built-in Plan Plugins:**

`ropt` includes default plugins providing common plan components:

*   **Steps** (via
    [`DefaultPlanStepPlugin`][ropt.plugins.plan.default.DefaultPlanStepPlugin]):
    *   `evaluator`: Performs ensemble evaluations
        ([`DefaultEnsembleEvaluatorStep`][ropt.plugins.plan.ensemble_evaluator.DefaultEnsembleEvaluatorStep]).
    *   `optimizer`: Runs an optimization algorithm using a configured optimizer
        plugin
        ([`DefaultOptimizerStep`][ropt.plugins.plan.optimizer.DefaultOptimizerStep]).
*   **Handlers** (via
    [`DefaultEventHandlerPlugin`][ropt.plugins.plan.default.DefaultEventHandlerPlugin]):
    *   `tracker`: Tracks the 'best' or 'last' valid result based on objective
        value and constraints
        ([`DefaultTrackerHandler`][ropt.plugins.plan._tracker.DefaultTrackerHandler]).
    *   `store`: Accumulates all results from specified sources
        ([`DefaultStoreHandler`][ropt.plugins.plan._store.DefaultStoreHandler]).
*   **Evaluators** (via
    [`DefaultEvaluatorPlugin`][ropt.plugins.plan.default.DefaultEvaluatorPlugin]):
    *   `function_evaluator`: Forwards calculations to a given evaluation
        function
        ([`DefaultFunctionEvaluator`][ropt.plugins.plan._function_evaluator.DefaultFunctionEvaluator]).

These built-in components allow for the construction of standard optimization
workflows out-of-the-box, while the plugin architecture enables customization
and extension.
"""

from .base import EventHandler, EventHandlerPlugin, PlanStep, PlanStepPlugin

__all__ = [
    "EventHandler",
    "EventHandlerPlugin",
    "PlanStep",
    "PlanStepPlugin",
]
