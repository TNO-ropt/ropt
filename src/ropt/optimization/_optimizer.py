"""Ensemble optimizer class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import numpy as np

from ropt.enums import ConstraintType, OptimizerExitCode
from ropt.exceptions import OptimizationAborted
from ropt.results import FunctionResults, GradientResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.ensemble_evaluator import EnsembleEvaluator
    from ropt.plugins import PluginManager
    from ropt.plugins.optimizer.base import Optimizer
    from ropt.results import Functions, Gradients, Results


class SignalEvaluationCallback(Protocol):
    """Protocol for a callback signaling the occurrence of an evaluation.

    This callback is invoked whenever an evaluation occurs, enabling custom
    handling or tracking of evaluation events.
    """

    def __call__(self, results: Optional[Tuple[Results, ...]] = None, /) -> None:
        """Callback protocol for signaling the start and end of evaluations.

        When provided to an ensemble optimizer, this callback is invoked both
        before and after each evaluation. Prior to the start of an evaluation,
        this method is called with `results` set to `None`. Upon completion of
        the evaluation, it is called with `results` containing the evaluation
        output.

        Args:
            results: The results produced by the evaluation, or `None` if the
                     evaluation has not yet completed.
        """


class NestedOptimizerCallback(Protocol):
    """Protocol for functions that start a nested optimization."""

    def __call__(
        self, variables: NDArray[np.float64], /
    ) -> Tuple[Optional[FunctionResults], bool]:
        """Callback protocol for executing a nested optimization.

        This function is called during each function evaluation within an
        ensemble optimizer to initiate a nested optimization process. It
        takes the current variables as arguments and returns a tuple containing
        the result of the nested optimization (or `None` if no result is
        available) and a boolean indicating whether the optimization was
        aborted by the user.

        Args:
            variables: The current variables to initialize the nested optimization.

        Returns:
            The result of the nested optimization and a boolean indicating abortion.
        """


class EnsembleOptimizer:
    """Optimizer class for running ensemble optimizations."""

    def __init__(
        self,
        enopt_config: EnOptConfig,
        ensemble_evaluator: EnsembleEvaluator,
        plugin_manager: PluginManager,
        signal_evaluation: Optional[SignalEvaluationCallback] = None,
        nested_optimizer: Optional[NestedOptimizerCallback] = None,
    ) -> None:
        """Initialize the ensemble optimizer class.

        This class requires at least three arguments that together define a
        single optimization run:

        1. An [`EnOptConfig`][ropt.config.enopt.EnOptConfig] that contains all
           configuration settings for the optimization.
        2. An [`EnsembleEvaluator`][ropt.ensemble_evaluator.EnsembleEvaluator]
           object responsible for evaluating functions.
        3. A [`PluginManager`][ropt.plugins.PluginManager] object that provides
           access to optimizer plugins.

        Additionally, two optional callbacks can be provided:

        1. A [`SignalEvaluationCallback`][ropt.optimization.SignalEvaluationCallback]
           that is called before and after each function evaluation.
        2. A [`NestedOptimizerCallback`][ropt.optimization.NestedOptimizerCallback]
           that is called at each function evaluation to run a nested optimization.

        Args:
            enopt_config:       The ensemble optimization configuration.
            ensemble_evaluator: The evaluator object for function evaluations.
            plugin_manager:     The plugin manager for accessing optimizer plugins.
            signal_evaluation:  Optional callback to signal evaluations.
            nested_optimizer:   Optional callback for running nested optimizations.
        """
        self._enopt_config = enopt_config
        self._function_evaluator = ensemble_evaluator
        self._plugin_manager = plugin_manager
        self._signal_evaluation = signal_evaluation
        self._nested_optimizer = nested_optimizer

        # This stores the values of the fixed variable
        self._fixed_variables: NDArray[np.float64]

        # For implementing max_functions:
        self._completed_functions = 0

        # Whether NaN values are allowed:
        self._allow_nan = False
        self._optimizer: Optimizer = self._plugin_manager.get_plugin(
            "optimizer", method=self._enopt_config.optimizer.method
        ).create(self._enopt_config, self._optimizer_callback)
        self._allow_nan = self._optimizer.allow_nan

    @property
    def is_parallel(self) -> bool:
        """Check if the optimization is parallelized.

        Returns:
            True if the optimization is parallelized; otherwise, False.
        """
        return self._optimizer.is_parallel

    def start(self, variables: NDArray[np.float64]) -> OptimizerExitCode:
        """Start the optimization.

        Args:
            variables: The initial variables for the optimization.

        Returns:
            An exit code indicating the reason for termination.
        """
        self._fixed_variables = variables.copy()
        exit_code = OptimizerExitCode.OPTIMIZER_STEP_FINISHED
        try:
            self._optimizer.start(variables)
        except OptimizationAborted as exc:
            exit_code = exc.exit_code
        return exit_code

    def _optimizer_callback(
        self,
        variables: NDArray[np.float64],
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert return_functions or return_gradients

        self._check_stopping_criteria()

        variables = self._get_completed_variables(variables)

        # Run any nested steps, when this improves the objective, this may
        # change the fixed variables and the current optimal result:
        if self._nested_optimizer is not None:
            # Nested optimization does not support parallel # evaluation:
            assert variables.ndim == 1
            nested_results, aborted = self._nested_optimizer(variables)
            if aborted:
                raise OptimizationAborted(exit_code=OptimizerExitCode.USER_ABORT)
            if nested_results is None:
                raise OptimizationAborted(
                    exit_code=OptimizerExitCode.NESTED_OPTIMIZER_FAILED
                )
            variables = (
                nested_results.evaluations.variables
                if nested_results.evaluations.scaled_variables is None
                else nested_results.evaluations.scaled_variables
            )
            self._fixed_variables = variables.copy()

        results = self._run_evaluations(
            variables,
            compute_functions=return_functions,
            compute_gradients=return_gradients,
        )

        # Functions and gradients might need to be scaled:
        scales, offsets = self._get_scale_parameters()

        functions = np.array([])
        if return_functions:
            # Functions might be parallelized hence we need potentially to
            # process a list of function results:
            functions_list = [
                self._scale_constraints(
                    self._functions_from_results(item.functions), scales, offsets
                )
                for item in results
                if isinstance(item, FunctionResults)
            ]
            self._completed_functions += len(functions_list)
            functions = (
                np.vstack(functions_list) if variables.ndim > 1 else functions_list[0]
            )

        gradients = np.array([])
        if return_gradients:
            # Gradients cannot be parallelized, there is at most one gradient:
            gradients = self._scale_constraints(
                self._gradients_from_results(
                    next(
                        item.gradients
                        for item in results
                        if isinstance(item, GradientResults)
                    ),
                    self._enopt_config.variables.indices,
                ),
                scales,
            )

        return functions, gradients

    def _get_completed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        indices = self._enopt_config.variables.indices
        if indices is not None:
            if variables.ndim > 1:
                tmp_variables = np.repeat(
                    self._fixed_variables[np.newaxis, :], variables.shape[0], axis=0
                )
                tmp_variables[:, indices] = variables
            else:
                tmp_variables = self._fixed_variables.copy()
                tmp_variables[indices] = variables
            return tmp_variables
        return variables.copy()

    def _check_stopping_criteria(self) -> None:
        max_functions = self._enopt_config.optimizer.max_functions
        if max_functions is not None and self._completed_functions >= max_functions:
            raise OptimizationAborted(exit_code=OptimizerExitCode.MAX_FUNCTIONS_REACHED)

    def _run_evaluations(
        self,
        variables: NDArray[np.float64],
        *,
        compute_functions: bool = False,
        compute_gradients: bool = False,
    ) -> Tuple[Results, ...]:
        assert compute_functions or compute_gradients
        if self._signal_evaluation:
            self._signal_evaluation()
        results = self._function_evaluator.calculate(
            variables,
            compute_functions=compute_functions,
            compute_gradients=compute_gradients,
        )
        if self._signal_evaluation:
            self._signal_evaluation(results)

        # If the configuration allows for zero successful realizations, there
        # will always be results. However, they may all be equal to `np.nan`. If
        # the optimizer does not set the allow_nan flag, it cannot handle such a
        # case, and we need to check for it:
        assert self._enopt_config.realizations.realization_min_success is not None
        check_failures = (
            self._enopt_config.realizations.realization_min_success < 1
            and not self._allow_nan
        )
        for result in results:
            assert isinstance(result, (FunctionResults, GradientResults))
            if (
                (isinstance(result, FunctionResults) and result.functions is None)
                or (isinstance(result, GradientResults) and result.gradients is None)
                or (check_failures and np.all(result.realizations.failed_realizations))
            ):
                raise OptimizationAborted(
                    exit_code=OptimizerExitCode.TOO_FEW_REALIZATIONS
                )

        return results

    @staticmethod
    def _functions_from_results(functions: Optional[Functions]) -> NDArray[np.float64]:
        assert functions is not None
        return (
            np.array(functions.weighted_objective, ndmin=1)
            if functions.constraints is None
            else np.append(functions.weighted_objective, functions.constraints)
        )

    @staticmethod
    def _gradients_from_results(
        gradients: Optional[Gradients], variable_indices: Optional[NDArray[np.intc]]
    ) -> NDArray[np.float64]:
        assert gradients is not None
        weighted_objective_gradient = (
            gradients.weighted_objective.copy()
            if variable_indices is None
            else gradients.weighted_objective[variable_indices]
        )
        constraint_gradients = (
            None
            if gradients.constraints is None
            else (
                gradients.constraints.copy()
                if variable_indices is None
                else gradients.constraints[:, variable_indices]
            )
        )
        return (
            np.expand_dims(weighted_objective_gradient, axis=0)
            if constraint_gradients is None
            else np.vstack((weighted_objective_gradient, constraint_gradients))
        )

    def _get_constraint_scales(self, config: EnOptConfig) -> NDArray[np.float64]:
        assert config.nonlinear_constraints is not None
        scales = self._function_evaluator.constraint_auto_scales
        if scales is None:
            return config.nonlinear_constraints.scales
        return config.nonlinear_constraints.scales * scales

    def _get_scale_parameters(
        self,
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        if self._enopt_config.nonlinear_constraints is not None:
            offsets = self._enopt_config.nonlinear_constraints.rhs_values
            scales = self._get_constraint_scales(self._enopt_config)
            scales = np.where(
                self._enopt_config.nonlinear_constraints.types == ConstraintType.GE,
                -scales,
                scales,
            )
            return scales, offsets
        return None, None

    def _scale_constraints(
        self,
        functions: NDArray[np.float64],
        scales: Optional[NDArray[np.float64]],
        offsets: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if functions.size > 1:
            if offsets is not None:
                functions[1:, ...] = functions[1:, ...] - offsets
            if scales is not None:
                if functions.ndim > 1:
                    scales = scales[:, np.newaxis]
                functions[1:, ...] = functions[1:, ...] / scales
        return functions
