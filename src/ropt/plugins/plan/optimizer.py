"""This module implements the default optimizer step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002

from ropt.config.enopt import EnOptConfig
from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, OptimizerExitCode
from ropt.optimization import EnsembleOptimizer
from ropt.plan import Event, Plan
from ropt.plugins.plan.base import PlanStep
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.results import Results
    from ropt.transforms import OptModelTransforms


MetaDataType = dict[str, int | float | bool | str]


class DefaultOptimizerStep(PlanStep):
    """The default optimizer step.

    The optimizer step performs an optimization, yielding a sequence of
    [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults] objects. The optimizer is
    configured using an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object or
    a dictionary that can be parsed into such an object. While the initial values
    for optimization are typically specified in the configuration, they can be
    overridden by providing them directly.

    The optimizer step emits several signals:

    - [`START_OPTIMIZER_STEP`][ropt.enums.EventType.START_OPTIMIZER_STEP]:
      Emitted before the optimization starts.
    - [`FINISHED_OPTIMIZER_STEP`][ropt.enums.EventType.FINISHED_OPTIMIZER_STEP]:
      Emitted after the optimization finishes.
    - [`START_EVALUATION`][ropt.enums.EventType.START_EVALUATION]: Emitted
      before a function or gradient evaluation.
    - [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION]: Emitted
      after a function or gradient evaluation.

      The `FINISHED_EVALUATION` signal is particularly important as it passes
      the generated [`Results`][ropt.results.Results] objects. Result handlers
      specified in the plan will respond to this signal to process those results.

    The optimizer step supports nested optimizations, where each function
    evaluation in the optimization calls a function that should run the nested
    optimization and produce the result for the function evaluation.
    """

    def __init__(
        self,
        plan: Plan,
    ) -> None:
        """Initialize a default optimizer step.

        Args:
            plan: The plan that runs this step.
        """
        super().__init__(plan)

    def run(  # type: ignore[override]
        self,
        config: dict[str, Any] | EnOptConfig,
        transforms: OptModelTransforms | None = None,
        variables: ArrayLike | None = None,
        nested_optimization: Plan | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizerExitCode:
        """Run the optimizer step.

        Args:
            config:              The optimizer configuration.
            transforms:          Optional transforms object.
            variables: Variables to evaluate.
            nested_optimization: Optional nested plan.
            metadata:            Optional metadata to add to events.
        """
        self._config = EnOptConfig.model_validate(config, context=transforms)
        self._transforms = transforms
        self._nested_optimization = nested_optimization
        self._metadata = metadata

        self.emit_event(
            Event(
                event_type=EventType.START_OPTIMIZER_STEP,
                config=self._config,
                source=self.id,
            )
        )

        if variables is None:
            variables = self._config.variables.initial_values
        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)

        ensemble_evaluator = EnsembleEvaluator(
            self._config,
            self._transforms,
            self.plan.optimizer_context.evaluator,
            self.plan.optimizer_context.plugin_manager,
        )

        ensemble_optimizer = EnsembleOptimizer(
            enopt_config=self._config,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self.plan.optimizer_context.plugin_manager,
            nested_optimizer=(
                self._run_nested_plan if self._nested_optimization is not None else None
            ),
            signal_evaluation=self._signal_evaluation,
        )

        if ensemble_optimizer.is_parallel and self._nested_optimization is not None:
            msg = "Nested optimization detected: parallel evaluation not supported. "
            raise RuntimeError(msg)

        exit_code = ensemble_optimizer.start(variables)

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

        self.emit_event(
            Event(
                event_type=EventType.FINISHED_OPTIMIZER_STEP,
                config=self._config,
                source=self.id,
            )
        )

        return exit_code

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
        """
        self.plan.emit_event(event)

    def _signal_evaluation(self, results: tuple[Results, ...] | None = None) -> None:
        if results is None:
            self.emit_event(
                Event(
                    event_type=EventType.START_EVALUATION,
                    config=self._config,
                    source=self.id,
                )
            )
        else:
            if self._metadata is not None:
                for item in results:
                    item.metadata = deepcopy(self._metadata)

            data: dict[str, Any] = {}
            if self._transforms is not None:
                data["transformed_results"] = results
                data["results"] = [
                    item.transform_from_optimizer(self._transforms) for item in results
                ]
            else:
                data["results"] = results
            self.emit_event(
                Event(
                    event_type=EventType.FINISHED_EVALUATION,
                    config=self._config,
                    source=self.id,
                    data=data,
                ),
            )

    def _run_nested_plan(
        self, variables: NDArray[np.float64]
    ) -> tuple[FunctionResults | None, bool]:
        if self._nested_optimization is None:
            return None, False
        self._nested_optimization.set_parent(self.plan)
        results = self._nested_optimization.run_function(variables)
        if self._nested_optimization.aborted:
            self.plan.abort()
        if not isinstance(results, FunctionResults):
            msg = "Nested optimization must return a FunctionResults object."
            raise TypeError(msg)
        return results, self._nested_optimization.aborted
