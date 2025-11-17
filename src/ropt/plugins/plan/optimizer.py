"""This module implements the default optimizer step."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray  # noqa: TC002

from ropt.ensemble_evaluator import EnsembleEvaluator
from ropt.enums import EventType, ExitCode
from ropt.optimization import EnsembleOptimizer
from ropt.plan import Event, Plan
from ropt.plugins.plan.base import Evaluator, PlanStep
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.config import EnOptConfig
    from ropt.results import Results
    from ropt.transforms import OptModelTransforms


MetaDataType = dict[str, int | float | bool | str]


class NestedOptimizationCallable(Protocol):
    """Protocol for nested optimizer function calls."""

    def __call__(
        self, variables: NDArray[np.float64], /
    ) -> tuple[FunctionResults | None, bool]:
        """Run a nested optimization using the given variables.

        This functions defines the signature of the callable that defines a
        nested optimization in a
        [`DefaultOptimizerStep`][ropt.plugins.plan.optimizer.DefaultOptimizerStep].

        It accepts the variables to evaluate and returns a tuple. The first
        element is a `FunctionResults` object containing the results of the
        nested optimization, or `None` if the evaluation failed. The second
        element is a boolean flag indicating whether the optimization was
        aborted by the user.

        Args:
            variables: The matrix of variables to start the nested optimization.

        Returns:
            A tuple with the function results object, and a flag indicating
            whether the optimization was aborted.
        """


class DefaultOptimizerStep(PlanStep):
    """The default optimizer step for optimization plans.

    This step executes an optimization algorithm based on a provided
    configuration ([`EnOptConfig`][ropt.config.EnOptConfig] or a compatible
    dictionary). It iteratively performs function and potentially gradient
    evaluations, yielding a sequence of
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
      under the key `"results"`. Event handlers typically listen for this event
      to process or track optimization progress.
    - [`FINISHED_OPTIMIZER_STEP`][ropt.enums.EventType.FINISHED_OPTIMIZER_STEP]:
      Emitted after the entire optimization process concludes (successfully,
      or due to termination conditions or errors).

    This step also supports **nested optimization**. If a `nested_optimization`
    function is provided to the `run` method, the optimizer will execute a
    nested optimization at as part of each function evaluation. Each nested
    optimization run is done by creating a new plan. The provided function is
    then executed, passing the new plan and the variables. The
    `nested_optimization` function is expected to return a single
    [`FunctionResults`][ropt.results.FunctionResults] object.
    """

    def __init__(self, plan: Plan, *, evaluator: Evaluator) -> None:
        """Initialize a default optimizer step.

        Args:
            plan:      The plan that runs this step.
            evaluator: The evaluator object to run function evaluations.
        """
        super().__init__(plan)
        self._evaluator = evaluator

    def run(
        self,
        config: EnOptConfig,
        variables: ArrayLike,
        *,
        transforms: OptModelTransforms | None = None,
        nested_optimization: NestedOptimizationCallable | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExitCode:
        """Run the optimizer step to perform an optimization.

        This method executes the core logic of the optimizer step. It requires
        an optimizer configuration
        ([`EnOptConfig`][ropt.config.EnOptConfig]) and optionally accepts
        specific initial variable vectors, and/or a nested optimization plan,
        and metadata.

        If `variables` are not provided, the initial values specified in the
        `config` are used. If `variables` are provided, they override the
        config's initial values.

        If `metadata` is provided, it is attached to the
        [`Results`][ropt.results.Results] objects emitted via the
        `FINISHED_EVALUATION` event.

        If a `nested_optimization` callable is provided, a fresh plan will be
        constructed, and the callable will be called passing that plan and the
        initial variables to use. The callable should return a a single
        [`FunctionResults`][ropt.results.FunctionResults] object that should
        contain the results of the nested optimization.

        Args:
            config:              Optimizer configuration.
            transforms:          Optional transforms to apply to the variables,
                                 objectives, and constraints.
            variables:           Optional initial variable vector(s) to start from.
            nested_optimization: Optional callable to run a nested plan.
            metadata:            Optional dictionary to attach to emitted `Results`.

        Returns:
            An exit code indicating the outcome of the optimization.
        """
        self._config = config
        self._transforms = transforms
        self._nested_optimization = nested_optimization
        self._metadata = metadata

        event_data: dict[str, Any] = {"config": config, "transforms": transforms}

        self._emit_event(
            Event(event_type=EventType.START_OPTIMIZER_STEP, data=event_data)
        )

        variables = np.array(np.asarray(variables, dtype=np.float64), ndmin=1)
        if variables.shape != (self._config.variables.variable_count,):
            msg = "The input variables have the wrong shape"
            raise ValueError(msg)
        if self._transforms is not None and self._transforms.variables is not None:
            variables = self._transforms.variables.to_optimizer(variables)

        ensemble_evaluator = EnsembleEvaluator(
            self._config,
            self._transforms,
            self._evaluator.eval,
            self.plan.plugin_manager,
        )

        ensemble_optimizer = EnsembleOptimizer(
            enopt_config=self._config,
            transforms=self._transforms,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self.plan.plugin_manager,
            nested_optimizer=(
                self._run_nested_optimization
                if self._nested_optimization is not None
                else None
            ),
            signal_evaluation=self._signal_evaluation,
        )

        if ensemble_optimizer.is_parallel and self._nested_optimization is not None:
            msg = "Nested optimization detected: parallel evaluation not supported. "
            raise RuntimeError(msg)

        exit_code = ensemble_optimizer.start(variables)

        self._emit_event(
            Event(event_type=EventType.FINISHED_OPTIMIZER_STEP, data=event_data)
        )

        return exit_code

    def _emit_event(self, event: Event) -> None:
        for handler in self.event_handlers:
            if event.event_type in handler.event_types:
                handler.handle_event(event)

    def _signal_evaluation(self, results: tuple[Results, ...] | None = None) -> None:
        event_data: dict[str, Any] = {
            "config": self._config,
            "transforms": self._transforms,
        }
        if results is None:
            self._emit_event(
                Event(event_type=EventType.START_EVALUATION, data=event_data)
            )
        else:
            if self._metadata is not None:
                for item in results:
                    item.metadata = deepcopy(self._metadata)

            event_data["results"] = results
            self._emit_event(
                Event(event_type=EventType.FINISHED_EVALUATION, data=event_data),
            )

    def _run_nested_optimization(
        self, variables: NDArray[np.float64]
    ) -> tuple[FunctionResults | None, bool]:
        if self._nested_optimization is None:
            return None, False
        results, aborted = self._nested_optimization(variables)
        if not isinstance(results, FunctionResults | None):
            msg = "Nested optimization must return a FunctionResults object or None ."
            raise TypeError(msg)
        return results, aborted
