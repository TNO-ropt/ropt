"""This module implements the default optimizer step."""

from __future__ import annotations

from functools import partial
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

from ._utils import _get_set

if TYPE_CHECKING:
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
        *,
        config: dict[str, Any] | EnOptConfig,
        transforms: OptModelTransforms | None = None,
        tags: str | set[str] | None = None,
        nested_optimization: Plan | None = None,
    ) -> None:
        """Initialize a default optimizer step.

        Args:
            plan:                The plan that runs this step.
            config:              The optimizer configuration.
            transforms:          Optional transforms object.
            tags:                Tags to add to the emitted events.
            nested_optimization: Optional nested plan.
        """
        super().__init__(plan)

        self._config = EnOptConfig.model_validate(config)
        self._transforms = transforms
        self._tags = _get_set(tags)
        self._nested_optimization = nested_optimization
        self._nested_run_index = 0
        self["exit_code"] = None

    def run(  # type: ignore[override]
        self,
        *,
        variables: FunctionResults | NDArray[np.float64] | list[float] | None = None,
    ) -> None:
        """Run the optimizer step."""
        match variables:
            case FunctionResults():
                variables = variables.evaluations.variables
            case np.ndarray() | list():
                variables = np.asarray(variables, dtype=np.float64)
            case None:
                variables = None
            case _:
                msg = "Invalid initial variables."
                raise ValueError(msg)

        if variables is None:
            variables = self._config.variables.initial_values
        self._run(self._config, self._transforms, variables)

    def _run(
        self,
        enopt_config: EnOptConfig,
        transforms: OptModelTransforms | None,
        variables: NDArray[np.float64],
    ) -> None:
        self.emit_event(
            Event(
                event_type=EventType.START_OPTIMIZER_STEP,
                config=enopt_config,
                tags=self._tags,
            )
        )

        ensemble_evaluator = EnsembleEvaluator(
            enopt_config,
            transforms,
            self.plan.optimizer_context.evaluator,
            self.plan.plan_id,
            self.plan.optimizer_context.plugin_manager,
        )

        ensemble_optimizer = EnsembleOptimizer(
            enopt_config=enopt_config,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self.plan.optimizer_context.plugin_manager,
            nested_optimizer=(
                self._run_nested_plan if self._nested_optimization is not None else None
            ),
            signal_evaluation=partial(
                self._signal_evaluation, enopt_config, transforms
            ),
        )

        if ensemble_optimizer.is_parallel and self._nested_optimization is not None:
            msg = "Nested optimization detected: parallel evaluation not supported. "
            raise RuntimeError(msg)

        exit_code = ensemble_optimizer.start(variables)
        self["exit_code"] = exit_code

        self.emit_event(
            Event(
                event_type=EventType.FINISHED_OPTIMIZER_STEP,
                config=enopt_config,
                tags=self._tags,
                data={"exit_code": exit_code},
            )
        )

        if exit_code == OptimizerExitCode.USER_ABORT:
            self.plan.abort()

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
        """
        self.plan.emit_event(event)

    def _signal_evaluation(
        self,
        enopt_config: EnOptConfig,
        transforms: OptModelTransforms | None,
        results: tuple[Results, ...] | None = None,
        *,
        exit_code: OptimizerExitCode | None = None,
    ) -> None:
        """Called before and after the optimizer finishes an evaluation.

        Before the evaluation starts, this method is called with the `results`
        argument set to `None`. When an evaluation is has finished, it is called
        with `results` set to the results of the evaluation.

        Args:
            enopt_config: The configuration object.
            transforms:   Optional transforms object.
            results:      The results produced by the evaluation.
            exit_code:    An exit code if that may be set if the evaluation completed.
        """
        if results is None:
            self.emit_event(
                Event(
                    event_type=EventType.START_EVALUATION,
                    config=enopt_config,
                    tags=self._tags,
                )
            )
        else:
            data: dict[str, Any] = {"exit_code": exit_code}
            if transforms is not None:
                data["transformed_results"] = results
                data["results"] = [
                    item.transform_from_optimizer(transforms) for item in results
                ]
            else:
                data["results"] = results
            self.emit_event(
                Event(
                    event_type=EventType.FINISHED_EVALUATION,
                    config=enopt_config,
                    tags=self._tags,
                    data=data,
                ),
            )

    def _run_nested_plan(
        self, variables: NDArray[np.float64]
    ) -> tuple[FunctionResults | None, bool]:
        """Run a  nested plan.

        Args:
            variables:     Variables to set in the nested plan.

        Returns:
            The variables generated by the nested plan.
        """
        if self._nested_optimization is None:
            return None, False
        self._nested_optimization.set_parent(self.plan, self._nested_run_index)
        self._nested_run_index += 1
        results = self._nested_optimization.run_function(variables)
        if self._nested_optimization.aborted:
            self.plan.abort()
        if not isinstance(results, FunctionResults):
            msg = "Nested optimization must return a FunctionResults object."
            raise TypeError(msg)
        return results, self._nested_optimization.aborted
