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
    """The default optimizer step for optimization plans.

    This step executes an optimization algorithm based on a provided
    configuration ([`EnOptConfig`][ropt.config.enopt.EnOptConfig] or a
    compatible dictionary). It iteratively performs function and potentially
    gradient evaluations, yielding a sequence of
    [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults] objects.

    While initial variable values are typically specified in the configuration,
    they can be overridden by passing them directly to the `run` method.

    The step emits the following events during its execution:

    - [`START_OPTIMIZER_STEP`][ropt.enums.EventType.START_OPTIMIZER_STEP]:
      Emitted just before the optimization process begins.
    - [`START_EVALUATION`][ropt.enums.EventType.START_EVALUATION]: Emitted
      immediately before an ensemble evaluation (for functions or gradients)
      is requested from the underlying optimizer.
    - [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION]: Emitted
      after an evaluation completes. This event carries the generated
      [`Results`][ropt.results.Results] object(s) in its `data` dictionary
      under the key `"results"`. Plan handlers typically listen for this event
      to process or track optimization progress.
    - [`FINISHED_OPTIMIZER_STEP`][ropt.enums.EventType.FINISHED_OPTIMIZER_STEP]:
      Emitted after the entire optimization process concludes (successfully,
      or due to termination conditions or errors).

    This step also supports **nested optimization**. If a `nested_optimization`
    plan is provided to the `run` method, the optimizer will execute this nested
    plan for each function evaluation instead of calling the standard ensemble
    evaluator. The nested plan is expected to return a single
    [`FunctionResults`][ropt.results.FunctionResults] object.
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

    def run(
        self,
        config: dict[str, Any] | EnOptConfig,
        transforms: OptModelTransforms | None = None,
        variables: ArrayLike | None = None,
        nested_optimization: Plan | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizerExitCode:
        """Run the optimizer step to perform an optimization.

        This method executes the core logic of the optimizer step. It requires
        an optimizer configuration
        ([`EnOptConfig`][ropt.config.enopt.EnOptConfig] or a compatible
        dictionary) and optionally accepts specific initial variable vectors,
        transforms, a nested optimization plan, and metadata.

        If `variables` are not provided, the initial values specified in the
        `config` are used. If `variables` are provided, they override the
        config's initial values.

        If a `transforms` object is given, it is passed to the optimizer to
        transform variables and results between user and optimizer domains (see
        [`ropt.transforms`][ropt.transforms]).

        If `metadata` is provided, it is attached to the
        [`Results`][ropt.results.Results] objects emitted via the
        `FINISHED_EVALUATION` event.

        If a `nested_optimization` plan is provided, it will be executed for
        each function evaluation instead of the standard ensemble evaluator.

        Args:
            config:              Optimizer configuration.
            transforms:          Optional transforms object.
            variables:           Optional initial variable vector(s) to start optimization from.
            nested_optimization: Optional nested plan.
            metadata:            Optional dictionary to attach to emitted `Results`.

        Returns:
            An [`OptimizerExitCode`][ropt.enums.OptimizerExitCode] indicating the outcome of the optimization.
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
        else:
            variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)
            if transforms is not None and transforms.variables is not None:
                variables = transforms.variables.to_optimizer(variables)

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

            data: dict[str, Any] = {"results": results}
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
