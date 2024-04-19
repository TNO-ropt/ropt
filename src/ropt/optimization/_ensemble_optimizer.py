from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, NoReturn, Optional, Sequence

from ropt.config.plan import PlanConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.plugins import PluginManager

from ._events import OptimizationEventBroker
from ._plan import Plan, PlanContext

if TYPE_CHECKING:
    from ropt.evaluator import Evaluator
    from ropt.events import OptimizationEvent
    from ropt.results import FunctionResults


class EnsembleOptimizer:
    """The Ensemble Optimizer Class.

    This class serves as the central component for executing an optimization
    plan. Upon initialization, an `EnsembleOptimizer` object accepts and stores
    a callback with the [`Evaluator`][ropt.evaluator.Evaluator] signature.
    Before initiating the optimization, additional callbacks can be incorporated
    using the [`add_observer`][ropt.optimization.EnsembleOptimizer.add_observer]
    method to respond to events occurring during the optimization. The
    optimization plan is initiated with the
    [`start_optimization`][ropt.optimization.EnsembleOptimizer.start_optimization]
    method. To prematurely halt the optimization workflow from within callback
    functions, the
    [`abort_optimization`][ropt.optimization.EnsembleOptimizer.abort_optimization]
    method is available. Upon the completion of the optimization process, the
    [`results`][ropt.optimization.EnsembleOptimizer.results] property can be
    used to examine any stored results.
    """

    def __init__(
        self, evaluator: Evaluator, plugin_manager: Optional[PluginManager] = None
    ) -> None:
        """Initialize an `EnsembleOptimizer` object.

        Args:
            evaluator:      An evaluator callback
            plugin_manager: Optional plugin manager
        """
        self._evaluator = evaluator
        self._events = OptimizationEventBroker()
        self._results: Dict[str, Optional[FunctionResults]] = {}
        self._plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )

    @property
    def results(self) -> Dict[str, Optional[FunctionResults]]:
        """Return the dictionary of results.

        Results may be stored during optimization and can be accessed by name
        via this property. Examples of such results include the optimal result
        encountered, but the exact nature of each result and the names to access
        them are determined by the optimization plan.
        """
        return self._results

    def add_observer(
        self, event: EventType, callback: Callable[[OptimizationEvent], None]
    ) -> None:
        """Add a callback for an event.

        Throughout the optimization process, various events are triggered,
        offering opportunities to observe progress and capture intermediate
        results. This method facilitates the connection of callbacks to specific
        events. The callback should accept a single
        [OptimizationEvent][ropt.events.OptimizationEvent] object and not return
        anything. The method can be called multiple times to add multiple
        callbacks to the same event, and each will be executed when that event
        occurs.

        Args:
            event:    The event to respond to.
            callback: The callback function to be connected.
        """
        self._events.add_observer(event, callback)

    def start_optimization(
        self, plan: Sequence[Dict[str, Any]], seed: Optional[int] = None
    ) -> Optional[FunctionResults]:
        """Start the optimization.

        This method executes the provide optimization plan. An optional seed can
        be provided to initialize the random number generator that can be used
        by the steps in the plan.

        Args:
            plan: The optimization plan to run
            seed: Optional seed of the random number generator

        Returns:
            The reason for the terminating the optimization.
        """
        plan_config = PlanConfig.model_validate(plan)
        context = PlanContext(self._events, self._evaluator, seed, self._plugin_manager)
        runner = Plan(plan_config, context)
        runner.run()
        self._results = runner.results
        return runner.final_result

    @staticmethod
    def abort_optimization() -> NoReturn:
        """Abort the current optimization run.

        This method can be called from within callbacks to interrupt the ongoing
        optimization plan. The exact point at which the optimization is aborted
        depends on the step in the plan that is executing at that point. For
        example, within a running optimizer, the process will be interrupted
        after completing the current function evaluation.
        """
        raise OptimizationAborted(exit_code=OptimizerExitCode.USER_ABORT)
