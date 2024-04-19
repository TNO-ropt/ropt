"""This module defines the protocol to be followed by optimization steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from ropt.config.plan import EvaluatorStepConfig
from ropt.enums import EventType, OptimizerExitCode
from ropt.evaluator import EnsembleEvaluator
from ropt.exceptions import ConfigError, OptimizationAborted
from ropt.optimization import EvaluatorStep, Plan
from ropt.results import FunctionResults

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.optimization import PlanContext


class DefaultEvaluatorStep(EvaluatorStep):
    """The default evaluator step."""

    def __init__(
        self, config: Dict[str, Any], context: PlanContext, plan: Plan
    ) -> None:
        """Initialize a default evaluator step.

        Args:
            config:  The configuration of the step
            context: Context in which the step runs
            plan:    The plan containing this step
        """
        # This class is likely to be derived. We use different scheme for naming
        # private attributes than usual, so the normal one could be used in
        # derived classes.
        self._private_context = context
        self._private_plan = plan
        self._private_config = EvaluatorStepConfig.model_validate(config)

    def run(self, variables: Optional[NDArray[np.float64]]) -> bool:
        """Run the evaluator step.

        Args:
            variables: Optional variables to start running with

        Returns:
            The Whether a user abort occurred.
        """
        if self._private_plan.enopt_config is None:
            msg = "Optimizer configuration missing"
            raise ConfigError(msg)

        self._private_context.events.emit(
            event_type=EventType.START_EVALUATOR_STEP,
            config=self._private_plan.enopt_config,
        )

        assert self._private_context.rng is not None
        ensemble_evaluator = EnsembleEvaluator(
            self._private_plan.enopt_config,
            self._private_context.evaluator,
            self._private_context.result_id_iter,
            self._private_context.rng,
            self._private_context.plugin_manager,
        )

        if variables is None:
            variables = self._private_plan.enopt_config.variables.initial_values

        exit_code = OptimizerExitCode.EVALUATION_STEP_FINISHED
        try:
            results = ensemble_evaluator.calculate(
                variables, compute_functions=True, compute_gradients=False
            )
        except OptimizationAborted as exc:
            exit_code = exc.exit_code

        if self._private_config.id is not None:
            for item in results:
                item.metadata["step_id"] = self._private_config.id

        results = self._private_plan.track_results(results, self._private_config.id)

        assert results
        assert isinstance(results[0], FunctionResults)
        if results[0].functions is None:
            exit_code = OptimizerExitCode.TOO_FEW_REALIZATIONS

        self._private_context.events.emit(
            event_type=EventType.FINISHED_EVALUATOR_STEP,
            config=self._private_plan.enopt_config,
            results=results,
            exit_code=exit_code,
        )

        if results[0] is not None:
            self.process(results[0])

        return exit_code == OptimizerExitCode.USER_ABORT

    def process(self, results: FunctionResults) -> None:
        """Process the results of the evaluation.

        This implementation does nothing, the results can also be accessed by
        connecting to the `OptimizerExitCode.TOO_FEW_REALIZATIONS` signal.
        However, it may be convenient to derive from this class, and provide an
        implementation of this method instead.

        Args:
            results: The results to process
        """
