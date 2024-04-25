from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from ropt.enums import ConstraintType, OptimizerExitCode
from ropt.exceptions import ConfigError, OptimizationAborted
from ropt.results import (
    BoundConstraints,
    FunctionResults,
    GradientResults,
    LinearConstraints,
    NonlinearConstraints,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.evaluator import EnsembleEvaluator
    from ropt.optimization import OptimizerStep
    from ropt.plugins import PluginManager
    from ropt.results import Functions, Gradients, Results


class Optimizer:
    def __init__(
        self,
        *,
        optimizer_step: OptimizerStep,
        enopt_config: EnOptConfig,
        ensemble_evaluator: EnsembleEvaluator,
        plugin_manager: PluginManager,
    ) -> None:
        self._enopt_config = enopt_config
        self._optimizer_step = optimizer_step
        self._function_evaluator = ensemble_evaluator
        self._plugin_manager = plugin_manager

        # This stores the values of the fixed variable
        self._fixed_variables: NDArray[np.float64]

        # For implementing max_functions:
        self._completed_functions = 0

    def start(self, variables: NDArray[np.float64]) -> OptimizerExitCode:
        self._fixed_variables = variables.copy()

        optimizer = self._plugin_manager.get_backend(
            "optimizer", self._enopt_config.optimizer.backend
        )(self._enopt_config, self._optimizer_callback)
        exit_code = OptimizerExitCode.OPTIMIZER_STEP_FINISHED
        try:
            optimizer.start(variables)
        except OptimizationAborted as exc:
            exit_code = exc.exit_code
        return exit_code

    def _optimizer_callback(
        self,
        variables: NDArray[np.float64],
        *,
        return_functions: bool,
        return_gradients: bool,
        allow_nan: bool = False,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if (
            self._enopt_config.realizations.realization_min_success < 1
            and not allow_nan
        ):
            msg = "Failed function evaluations by the optimizer"
            raise ConfigError(msg)

        assert return_functions or return_gradients

        self._check_stopping_criteria()

        variables = self._get_completed_variables(variables)

        # Run any nested steps, when this improves the objective, this may
        # change the fixed variables and the current optimal result:
        nested_results, aborted = self._optimizer_step.run_nested_plan(variables)
        if nested_results is not None:
            if aborted:
                raise OptimizationAborted(exit_code=OptimizerExitCode.USER_ABORT)
            variables = nested_results.evaluations.variables.copy()
            self._fixed_variables = variables.copy()

        # TODO: After a nested step, we may be able to re-use its result:
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
                    variables[np.newaxis, :], variables.shape[0], axis=0
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
        self._optimizer_step.start_evaluation()
        results = self._function_evaluator.calculate(
            variables,
            compute_functions=compute_functions,
            compute_gradients=compute_gradients,
        )
        results = self._augment_results(results)
        self._optimizer_step.finish_evaluation(results)
        for result in results:
            if (isinstance(result, FunctionResults) and result.functions is None) or (
                isinstance(result, GradientResults) and result.gradients is None
            ):
                raise OptimizationAborted(
                    exit_code=OptimizerExitCode.TOO_FEW_REALIZATIONS
                )
        return results

    def _augment_results(self, results: Tuple[Results, ...]) -> Tuple[Results, ...]:
        for result in results:
            if isinstance(result, FunctionResults):
                result.bound_constraints = BoundConstraints.create(
                    self._enopt_config, result.evaluations
                )
                result.linear_constraints = LinearConstraints.create(
                    self._enopt_config, result.evaluations
                )
                result.nonlinear_constraints = NonlinearConstraints.create(
                    self._enopt_config,
                    result.functions,
                    self._function_evaluator.constraint_auto_scales,
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
        return (
            np.expand_dims(weighted_objective_gradient, axis=0)
            if gradients.constraints is None
            else np.vstack((weighted_objective_gradient, gradients.constraints))
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
