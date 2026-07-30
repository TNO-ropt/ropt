"""This module implements the default optimizer compute step."""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._logging import get_logger
from ropt.core import EnsembleEvaluator, EnsembleOptimizer
from ropt.enums import EnOptEventType, ExitCode
from ropt.events import EnOptEvent
from ropt.exceptions import Abort
from ropt.workflow.executors._worker import is_worker_process

from .base import ComputeStep

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ropt.context import EnOptContext
    from ropt.results import Results
    from ropt.workflow.evaluators import Evaluator


_logger = get_logger(__name__)

MetaDataType = dict[str, int | float | bool | str]


class OptimizationStep(ComputeStep):
    """The default optimizer compute step.

    Executes an optimization algorithm, iteratively performing function and
    gradient evaluations. Emits `START_OPTIMIZER`, `START_EVALUATION`,
    `FINISHED_EVALUATION`, and `FINISHED_OPTIMIZER` events.

    See [Optimization Workflows](../usage/workflows.md#events-emitted-by-optimizationstep)
    for the full event lifecycle description.
    """

    def __init__(self, *, evaluator: Evaluator) -> None:
        """Initialize a default optimizer.

        Args:
            evaluator: The evaluator object to run function evaluations.
        """
        super().__init__()
        self._evaluator = evaluator
        self._abort = threading.Event()
        self._running = False
        self._run_lock = threading.Lock()

    def run(
        self,
        context: EnOptContext,
        variables: ArrayLike,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExitCode:
        """Run the optimization.

        Args:
            context:    The optimizer context.
            variables:  Initial variable vector(s).
            metadata:   Optional dictionary attached to emitted
                [`Results`][ropt.results.Results] via the `FINISHED_EVALUATION`
                event.

        Returns:
            An exit code describing the outcome of the optimization.

        Raises:
            RuntimeError: If this step is already running on another thread.
            ValueError: If the input variables have the wrong shape.
        """
        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)
        if variables.shape != (context.variables.variable_count,):
            msg = "The input variables have the wrong shape"
            raise ValueError(msg)
        with self._run_lock:
            if self._running:
                msg = "The optimization step is already running on another thread."
                raise RuntimeError(msg)
            self._running = True
        try:
            return self._run(context, variables, metadata=metadata)
        finally:
            with self._run_lock:
                self._running = False

    def _run(
        self,
        context: EnOptContext,
        variables: NDArray[np.float64],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExitCode:
        context.lock()

        self._abort.clear()
        self._context = context
        self._metadata = metadata

        _logger.debug(
            "Configuration: %d variable(s), %d realization(s), %d objective(s)",
            context.variables.variable_count,
            context.realizations.weights.size,
            context.objectives.weights.size,
        )
        _logger.info("Starting optimization")
        self._emit_event(
            EnOptEvent(event_type=EnOptEventType.START_OPTIMIZER, context=context)
        )

        for transform in context.variable_transforms:
            variables = transform.to_optimizer(variables)

        ensemble_evaluator = EnsembleEvaluator(
            self._context,
            self._evaluator.eval,
        )
        ensemble_optimizer = EnsembleOptimizer(
            context=self._context,
            ensemble_evaluator=ensemble_evaluator,
            signal_evaluation=self._signal_evaluation,
        )
        exit_code = ensemble_optimizer.start(variables)

        _logger.info("Optimization finished: %s", exit_code.name)
        self._emit_event(
            EnOptEvent(event_type=EnOptEventType.FINISHED_OPTIMIZER, context=context)
        )

        return exit_code

    def abort(self) -> None:
        """Request a cooperative abort of the running optimization.

        Calling this method signals the optimization started by
        [`run`][ropt.workflow.compute_steps.OptimizationStep.run] to stop at the
        next evaluation boundary, causing `run` to return
        [`ExitCode.USER_ABORT`][ropt.enums.ExitCode]. The optimization is not
        interrupted mid-evaluation; the request takes effect before the next
        batch of function evaluations starts.

        This method is safe to call from any thread, for example from within an
        event handler observing the optimization. It only affects an
        optimization whose driver runs in the current process: an optimization
        running behind a process or HPC boundary cannot be reached and can only
        be terminated by stopping that process.

        See [Optimization Workflows](../usage/workflows.md#aborting-an-optimization)
        for details.
        """
        self._abort.set()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_abort", None)
        state.pop("_run_lock", None)
        state.pop("_running", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if is_worker_process():
            msg = (
                "An OptimizationStep cannot be transferred into a worker process; "
                "create it inside the worker instead."
            )
            raise RuntimeError(msg)
        self.__dict__.update(state)
        self._abort = threading.Event()
        self._running = False
        self._run_lock = threading.Lock()

    def _emit_event(self, event: EnOptEvent) -> None:
        for handler in self.event_handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)

    def _signal_evaluation(self, results: tuple[Results, ...] | None = None) -> None:
        if results is None:
            self._emit_event(
                EnOptEvent(
                    event_type=EnOptEventType.START_EVALUATION, context=self._context
                )
            )
            if self._abort.is_set():
                raise Abort(ExitCode.USER_ABORT)
        else:
            if self._metadata is not None:
                for item in results:
                    item.metadata = deepcopy(self._metadata)

            self._emit_event(
                EnOptEvent(
                    event_type=EnOptEventType.FINISHED_EVALUATION,
                    context=self._context,
                    results=results,
                ),
            )
