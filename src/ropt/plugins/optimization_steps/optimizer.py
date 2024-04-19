"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ropt.config.plan import OptimizerStepConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.evaluator import EnsembleEvaluator
from ropt.exceptions import ConfigError
from ropt.optimization import Optimizer, OptimizerStep, Plan
from ropt.utils import scaling

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.optimization import PlanContext
    from ropt.results import FunctionResults, Results


class DefaultOptimizerStep(OptimizerStep):
    """The default optimizer step."""

    def __init__(
        self, config: Dict[str, Any], context: PlanContext, plan: Plan
    ) -> None:
        """Initialize a default optimizer step.

        Args:
            config:  The configuration of the step
            context: Context in which the step runs
            plan:    The plan containing this step
        """
        self._config = OptimizerStepConfig.model_validate(config)
        self._context = context
        self._plan = plan

        self._nested_plan: Optional[Plan] = None
        if self._config.nested_plan is not None:
            self._nested_plan = Plan(self._config.nested_plan, self._context)

    def run(self, variables: Optional[NDArray[np.float64]]) -> bool:
        """Run the optimizer step.

        Args:
            variables: Optional variables to start running with

        Returns:
            Wether a user abort occurred.
        """
        if self._plan.enopt_config is None:
            msg = "Optimizer configuration missing"
            raise ConfigError(msg)

        self._context.events.emit(
            event_type=EventType.START_OPTIMIZER_STEP,
            config=self._plan.enopt_config,
        )

        assert self._context.rng is not None
        ensemble_evaluator = EnsembleEvaluator(
            self._plan.enopt_config,
            self._context.evaluator,
            self._context.result_id_iter,
            self._context.rng,
            self._context.plugin_manager,
        )

        if variables is None:
            variables = self._plan.enopt_config.variables.initial_values
        exit_code = Optimizer(
            enopt_config=self._plan.enopt_config,
            optimizer_step=self,
            ensemble_evaluator=ensemble_evaluator,
            plugin_manager=self._context.plugin_manager,
        ).start(variables)

        self._context.events.emit(
            event_type=EventType.FINISHED_OPTIMIZER_STEP,
            config=self._plan.enopt_config,
            exit_code=exit_code,
        )

        return exit_code == OptimizerExitCode.USER_ABORT

    def start_evaluation(self) -> None:
        """Called before the optimizer starts an evaluation."""
        self._context.events.emit(
            event_type=EventType.START_EVALUATION,
            config=self._plan.enopt_config,
        )

    def finish_evaluation(self, results: Tuple[Results, ...]) -> None:
        """Called after the optimizer finishes an evaluation.

        Args:
            results: The results produced by the evaluation.
        """
        if self._config.id is not None:
            for item in results:
                item.metadata["step_id"] = self._config.id

        results = self._plan.track_results(results, self._config.id)

        self._context.events.emit(
            event_type=EventType.FINISHED_EVALUATION,
            config=self._plan.enopt_config,
            results=results,
        )

    def run_nested_plan(
        self, variables: NDArray[np.float64]
    ) -> Tuple[Optional[FunctionResults], bool]:
        """Run a nested plan from an optimization run.

        Args:
            plan:      The plan to run
            variables: The variables to run with

        Returns:
            The optimal result of the run and whether the run was aborted.
        """
        if self._nested_plan is None:
            return None, False
        assert self._plan.enopt_config is not None
        unscaled_variables = scaling.scale_back_variables(
            self._plan.enopt_config, variables, axis=-1
        )
        if unscaled_variables is not None:
            variables = unscaled_variables
        aborted = self._nested_plan.run(variables)

        return (
            self._nested_plan.final_result
            if self._config.nested_result is None
            else self._nested_plan.results[self._config.nested_result],
            aborted,
        )
