"""This module provides the default plugin implementations for steps and event handlers.

It defines `DefaultPlanStepPlugin` and `DefaultEventHandlerPlugin`, which serve
as factories for the built-in plan components, enabling the creation of
standard optimization plans out-of-the-box.

**Supported Components:**

- **Steps:**
    - `ensemble_evaluator`: Performs ensemble evaluations
        ([`DefaultEnsembleEvaluatorStep`][ropt.plugins.plan.ensemble_evaluator.DefaultEnsembleEvaluatorStep]).
    - `optimizer`: Runs an optimization algorithm using a configured optimizer
        plugin
        ([`DefaultOptimizerStep`][ropt.plugins.plan.optimizer.DefaultOptimizerStep]).
- **Handlers:**
    - `tracker`: Tracks the 'best' or 'last' valid result based on objective
        value and constraints
        ([`DefaultTrackerHandler`][ropt.plugins.plan._tracker.DefaultTrackerHandler]).
    - `store`: Accumulates all results from specified sources
        ([`DefaultStoreHandler`][ropt.plugins.plan._store.DefaultStoreHandler]).
    - `observer`: Listens for events from specified sources, and calls a
      callback for each event
        ([`DefaultObserverHandler`][ropt.plugins.plan._observer.DefaultObserverHandler]).
- **Evaluators:**
    - `function_evaluator`: Evaluator that forwards calculations to a given evaluation function.
      ([`DefaultFunctionEvaluator`][ropt.plugins.plan._function_evaluator.DefaultFunctionEvaluator])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from ._function_evaluator import DefaultFunctionEvaluator
from ._observer import DefaultObserverHandler
from ._store import DefaultStoreHandler
from ._tracker import DefaultTrackerHandler
from .base import EvaluatorPlugin, EventHandlerPlugin, PlanComponent, PlanStepPlugin
from .ensemble_evaluator import DefaultEnsembleEvaluatorStep
from .optimizer import DefaultOptimizerStep

if TYPE_CHECKING:
    from ropt.plan import Plan
    from ropt.plugins.plan.base import Evaluator, EventHandler, PlanStep

_STEP_OBJECTS: Final[dict[str, type[PlanStep]]] = {
    "ensemble_evaluator": DefaultEnsembleEvaluatorStep,
    "optimizer": DefaultOptimizerStep,
}

_EVENT_HANDLER_OBJECTS: Final[dict[str, type[EventHandler]]] = {
    "observer": DefaultObserverHandler,
    "tracker": DefaultTrackerHandler,
    "store": DefaultStoreHandler,
}

_EVALUATOR_OBJECTS: Final[dict[str, type[Evaluator]]] = {
    "function_evaluator": DefaultFunctionEvaluator,
}


class DefaultEventHandlerPlugin(EventHandlerPlugin):
    """The default plugin for creating built-in event handlers.

    This plugin acts as a factory for the standard `EventHandler`
    implementations provided by `ropt`. It allows the
    [`PluginManager`][ropt.plugins.PluginManager] to instantiate these event
    handlers when requested by a [`Plan`][ropt.plan.Plan].

    **Supported Handlers:**

    - `tracker`: Creates a
        [`DefaultTrackerHandler`][ropt.plugins.plan._tracker.DefaultTrackerHandler]
        instance, which tracks either the 'best' or 'last' valid result based on
        objective value and constraints.
    - `store`: Creates a
        [`DefaultStoreHandler`][ropt.plugins.plan._store.DefaultStoreHandler]
        instance, which accumulates all results received from specified sources.
    - `observer`: Creates a
        [`DefaultObserverHandler`][ropt.plugins.plan._observer.DefaultObserverHandler]
        instance, which calls a callback for each event received from specified
        sources.
    """

    @classmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        tags: set[str] | None = None,
        sources: set[PlanComponent | str] | None = None,
        **kwargs: dict[str, Any],
    ) -> EventHandler:
        """Create an event handler.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        _, _, name = name.lower().rpartition("/")
        obj = _EVENT_HANDLER_OBJECTS.get(name)
        if obj is not None:
            return obj(plan, tags, sources, **kwargs)

        msg = f"Unknown event handler object type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in _EVENT_HANDLER_OBJECTS


class DefaultPlanStepPlugin(PlanStepPlugin):
    """The default plugin for creating built-in plan steps.

    This plugin acts as a factory for the standard `PlanStep` implementations
    provided by `ropt`. It allows the
    [`PluginManager`][ropt.plugins.PluginManager] to instantiate these steps
    when requested by a [`Plan`][ropt.plan.Plan].

    **Supported Steps:**

    - `ensemble_evaluator`: Creates a
        [`DefaultEnsembleEvaluatorStep`][ropt.plugins.plan.ensemble_evaluator.DefaultEnsembleEvaluatorStep]
        instance, which performs ensemble evaluations.
    - `optimizer`: Creates a
        [`DefaultOptimizerStep`][ropt.plugins.plan.optimizer.DefaultOptimizerStep]
        instance, which runs an optimization algorithm using a configured
        optimizer plugin.
    """

    @classmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        tags: set[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> PlanStep:
        """Create a step.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        _, _, name = name.lower().rpartition("/")
        step_obj = _STEP_OBJECTS.get(name)
        if step_obj is not None:
            return step_obj(plan, tags, **kwargs)

        msg = f"Unknown step type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in _STEP_OBJECTS


class DefaultEvaluatorPlugin(EvaluatorPlugin):
    """The default plugin for creating evaluators.

    This plugin acts as a factory for the standard evaluator implementations
    provided by `ropt`. It allows the
    [`PluginManager`][ropt.plugins.PluginManager] to instantiate these steps
    when requested by a [`Plan`][ropt.plan.Plan].

    **Supported Evaluators:**

    - `function_evaluator`: Creates a
        [`DefaultFunctionEvaluator`][ropt.plugins.plan._function_evaluator.DefaultFunctionEvaluator]
        instance, which uses function calls to calculated individual objectives
        and constraints.
    """

    @classmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        tags: set[str] | None = None,
        clients: set[PlanComponent | str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Evaluator:
        """Create a step.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        _, _, name = name.lower().rpartition("/")
        evaluator_obj = _EVALUATOR_OBJECTS.get(name)
        if evaluator_obj is not None:
            return evaluator_obj(plan, tags, clients, **kwargs)

        msg = f"Unknown evaluator type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in _EVALUATOR_OBJECTS
