from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import default_rng

from ropt.results import (
    ConstraintInfo,
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Gradients,
    Realizations,
    Results,
)

from ._evaluator_results import (
    _FunctionEvaluatorResults,
    _get_active_realizations,
    _get_function_and_gradient_results,
    _get_function_results,
    _get_gradient_results,
)
from ._function import _calculate_estimated_constraints, _calculate_estimated_objectives
from ._gradient import (
    _calculate_estimated_constraint_gradients,
    _calculate_estimated_objective_gradients,
    _perturb_variables,
)
from ._utils import _get_failed_realizations

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.evaluator import Evaluator
    from ropt.plugins import PluginManager
    from ropt.plugins.function_estimator.base import FunctionEstimator
    from ropt.plugins.realization_filter.base import RealizationFilter
    from ropt.plugins.sampler.base import Sampler
    from ropt.transforms import OptModelTransforms


class EnsembleEvaluator:
    """Construct functions and gradients from an ensemble of functions.

    The `EnsembleEvaluator` class is responsible for calculating functions and
    gradients from an ensemble of functions. It leverages the settings defined
    in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] configuration object to
    guide the calculations.

    The core functionality relies on an [`Evaluator`][ropt.evaluator.Evaluator]
    callable, which is used to evaluate the individual functions within the
    ensemble. The evaluator provides the raw function values, which are then
    processed by the `EnsembleEvaluator` to produce the final function and
    gradient estimates.
    """

    def __init__(
        self,
        config: EnOptConfig,
        transforms: OptModelTransforms | None,
        evaluator: Evaluator,
        plugin_manager: PluginManager,
    ) -> None:
        """Initialize the EnsembleEvaluator.

        This method sets up the `EnsembleEvaluator` with the necessary
        configuration, evaluator, and plugins.

        The `config` object contains all the settings required for the ensemble
        evaluation, such as the number of realizations, the function estimators,
        and the gradient settings. The `transforms` object can be used to
        transform the variables, objectives, and constraints before or after the
        evaluation. The `evaluator` callable should conform to the
        [`Evaluator`][ropt.evaluator.Evaluator] protocol. The `plugin_manager`
        is used to load the realization filters, function estimators, and
        samplers.

        Args:
            config:         The configuration object.
            transforms:     Optional transforms object.
            evaluator:      The callable for evaluating individual functions.
            plugin_manager: A plugin manager to load required plugins.
        """
        self._config = config
        self._transforms = transforms
        self._evaluator = evaluator
        self._realization_filters = self._init_realization_filters(plugin_manager)
        self._function_estimators = self._init_function_estimators(plugin_manager)
        rng = default_rng(config.gradient.seed)
        self._samplers = self._init_samplers(rng, plugin_manager)
        self._cache_for_gradient: FunctionResults | None = None

    def calculate(
        self,
        variables: NDArray[np.float64],
        *,
        compute_functions: bool,
        compute_gradients: bool,
    ) -> tuple[Results, ...]:
        """Evaluate the given variable vectors.

        This method calculates functions, gradients, or both, based on the
        provided variable vectors and the specified flags.

        The `variables` argument can be a single vector or a matrix where each
        row is a variable vector.

        Args:
            variables:         The variable vectors to evaluate.
            compute_functions: Whether to calculate functions.
            compute_gradients: Whether to calculate gradients.

        Returns:
            The results for function evaluations and/or gradient evaluations.
        """
        assert compute_functions or compute_gradients

        # Only functions:
        if compute_functions and not compute_gradients:
            return self._calculate_functions(variables)

        # Parallel evaluation is not supported for gradient-based optimizers:
        assert variables.ndim == 1

        # Only a gradient, and there is a cached function value:
        if (
            compute_gradients
            and not compute_functions
            and self._cache_for_gradient is not None
            and np.allclose(
                self._cache_for_gradient.evaluations.variables,
                variables,
                rtol=0.0,
                atol=1e-15,
            )
        ):
            # This assumes that the cached function value is calculated with the
            # same parameters as the gradient is to be calculated. This is true
            # because each optimization step is a single optimizer run and uses
            # its own copy of an `EnsembleFunctionEvaluation` object.
            return self._calculate_gradients(variables, self._config.variables.mask)

        # A function + gradient, or a gradient without cached function:
        self._cache_for_gradient = None
        return self._calculate_both(variables, self._config.variables.mask)

    def _calculate_functions(
        self, variables: NDArray[np.float64]
    ) -> tuple[FunctionResults, ...]:
        if variables.ndim == 1:
            variables = variables[np.newaxis, :]
        active_objectives, active_constraints = _get_active_realizations(self._config)
        function_results = tuple(
            self._calculate_one_set_of_functions(f_eval_results, variables[idx, :])
            for idx, f_eval_results in _get_function_results(
                self._config,
                self._transforms,
                self._evaluator,
                variables,
                active_objectives,
                active_constraints,
            )
        )

        # Gradient-based methods are currently not parallelized, and there is
        # only one function result. That function may be needed in a gradient
        # calculation, so it is cached here:
        self._cache_for_gradient = function_results[0]

        return function_results

    def _calculate_one_set_of_functions(
        self, f_eval_results: _FunctionEvaluatorResults, variables: NDArray[np.float64]
    ) -> FunctionResults:
        (
            objective_weights,
            constraint_weights,
        ) = self._calculate_filtered_realization_weights(f_eval_results)

        assert self._config.gradient.perturbation_min_success is not None
        failed_realizations = _get_failed_realizations(
            f_eval_results.objectives,
            None,
            self._config.gradient.perturbation_min_success,
        )

        assert self._config.realizations.realization_min_success is not None
        if (
            np.count_nonzero(~failed_realizations)
            >= self._config.realizations.realization_min_success
        ):
            functions = self._compute_functions(
                f_eval_results.objectives,
                f_eval_results.constraints,
                objective_weights,
                constraint_weights,
                failed_realizations,
            )
        else:
            functions = None

        evaluations = FunctionEvaluations.create(
            variables=variables,
            objectives=f_eval_results.objectives,
            constraints=f_eval_results.constraints,
            evaluation_info=f_eval_results.evaluation_info,
        )

        return FunctionResults(
            batch_id=f_eval_results.batch_id,
            metadata={},
            evaluations=evaluations,
            realizations=Realizations(
                failed_realizations=failed_realizations,
                objective_weights=objective_weights,
                constraint_weights=constraint_weights,
            ),
            functions=functions,
            constraint_info=ConstraintInfo.create(
                self._config,
                evaluations.variables,
                functions.constraints if functions is not None else None,
            ),
        )

    def _calculate_gradients(
        self,
        variables: NDArray[np.float64],
        mask: NDArray[np.bool_] | None,
    ) -> tuple[GradientResults]:
        perturbed_variables = _perturb_variables(
            self._config, variables, self._samplers
        )

        # No functions are computed in this case, instead they must have been
        # computed in a previous run, with the results stored in the
        # cached_result argument. In that case, we can skip any realizations
        # that have a weight equal to zero, we we pull those from the cached
        # results.
        assert self._cache_for_gradient is not None
        assert self._cache_for_gradient.realizations is not None
        objective_weights = self._cache_for_gradient.realizations.objective_weights
        constraint_weights = self._cache_for_gradient.realizations.constraint_weights

        active_objectives, active_constraints = _get_active_realizations(
            self._config,
            objective_weights=objective_weights,
            constraint_weights=constraint_weights,
        )
        g_eval_results = _get_gradient_results(
            self._config,
            self._transforms,
            self._evaluator,
            perturbed_variables,
            active_objectives,
            active_constraints,
        )

        assert self._config.gradient.perturbation_min_success is not None
        failed_realizations = _get_failed_realizations(
            self._cache_for_gradient.evaluations.objectives,
            g_eval_results.perturbed_objectives,
            self._config.gradient.perturbation_min_success,
        )
        assert self._config.realizations.realization_min_success is not None
        if (
            np.count_nonzero(~failed_realizations)
            >= self._config.realizations.realization_min_success
        ):
            gradients = self._compute_gradients(
                variables,
                mask,
                perturbed_variables,
                self._cache_for_gradient.evaluations.objectives,
                self._cache_for_gradient.evaluations.constraints,
                g_eval_results.perturbed_objectives,
                g_eval_results.perturbed_constraints,
                objective_weights,
                constraint_weights,
                failed_realizations,
            )
        else:
            gradients = None

        assert g_eval_results.perturbed_objectives is not None
        return (
            GradientResults(
                batch_id=g_eval_results.batch_id,
                metadata={},
                evaluations=GradientEvaluations.create(
                    variables=variables,
                    perturbed_variables=perturbed_variables,
                    perturbed_objectives=g_eval_results.perturbed_objectives,
                    perturbed_constraints=g_eval_results.perturbed_constraints,
                    evaluation_info=g_eval_results.evaluation_info,
                ),
                realizations=Realizations(
                    failed_realizations=failed_realizations,
                    objective_weights=objective_weights,
                    constraint_weights=constraint_weights,
                ),
                gradients=gradients,
            ),
        )

    def _calculate_both(
        self,
        variables: NDArray[np.float64],
        mask: NDArray[np.bool_] | None,
    ) -> tuple[FunctionResults, GradientResults]:
        perturbed_variables = _perturb_variables(
            self._config, variables, self._samplers
        )
        active_objectives, active_constraints = _get_active_realizations(self._config)
        f_eval_results, g_eval_results = _get_function_and_gradient_results(
            self._config,
            self._transforms,
            self._evaluator,
            variables,
            perturbed_variables,
            active_objectives,
            active_constraints,
        )

        evaluations = FunctionEvaluations.create(
            variables=variables,
            objectives=f_eval_results.objectives,
            constraints=f_eval_results.constraints,
            evaluation_info=f_eval_results.evaluation_info,
        )

        (
            objective_weights,
            constraint_weights,
        ) = self._calculate_filtered_realization_weights(
            f_eval_results,
        )

        assert self._config.gradient.perturbation_min_success is not None
        failed_realizations = _get_failed_realizations(
            f_eval_results.objectives,
            None,
            self._config.gradient.perturbation_min_success,
        )
        assert self._config.realizations.realization_min_success is not None
        if (
            np.count_nonzero(~failed_realizations)
            >= self._config.realizations.realization_min_success
        ):
            functions = self._compute_functions(
                f_eval_results.objectives,
                f_eval_results.constraints,
                objective_weights,
                constraint_weights,
                failed_realizations,
            )
        else:
            functions = None

        function_results = FunctionResults(
            batch_id=f_eval_results.batch_id,
            metadata={},
            evaluations=evaluations,
            realizations=Realizations(
                failed_realizations=failed_realizations,
                objective_weights=objective_weights,
                constraint_weights=constraint_weights,
            ),
            functions=functions,
            constraint_info=ConstraintInfo.create(
                self._config,
                evaluations.variables,
                functions.constraints if functions is not None else None,
            ),
        )

        assert self._config.gradient.perturbation_min_success is not None
        failed_realizations = _get_failed_realizations(
            f_eval_results.objectives,
            g_eval_results.perturbed_objectives,
            self._config.gradient.perturbation_min_success,
        )
        assert self._config.realizations.realization_min_success is not None
        if (
            np.count_nonzero(~failed_realizations)
            >= self._config.realizations.realization_min_success
        ):
            gradients = self._compute_gradients(
                variables,
                mask,
                perturbed_variables,
                f_eval_results.objectives,
                f_eval_results.constraints,
                g_eval_results.perturbed_objectives,
                g_eval_results.perturbed_constraints,
                objective_weights,
                constraint_weights,
                failed_realizations,
            )
        else:
            gradients = None

        gradient_results = GradientResults(
            batch_id=g_eval_results.batch_id,
            metadata={},
            evaluations=GradientEvaluations.create(
                variables=variables,
                perturbed_variables=perturbed_variables,
                perturbed_objectives=g_eval_results.perturbed_objectives,
                perturbed_constraints=g_eval_results.perturbed_constraints,
                evaluation_info=g_eval_results.evaluation_info,
            ),
            realizations=Realizations(
                failed_realizations=failed_realizations,
                objective_weights=objective_weights,
                constraint_weights=constraint_weights,
            ),
            gradients=gradients,
        )

        return function_results, gradient_results

    def _compute_functions(
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
        objective_weights: NDArray[np.float64] | None,
        constraint_weights: NDArray[np.float64] | None,
        failed_realizations: NDArray[np.bool_],
    ) -> Functions:
        # Individual objective and constraint functions are calculated from the
        # realizations using one or more function estimators:
        if np.all(failed_realizations):
            objectives = np.empty(
                self._config.objectives.weights.size, dtype=np.float64
            )
            objectives.fill(np.nan)
            if constraints is not None:
                assert self._config.nonlinear_constraints is not None
                constraints = np.empty(
                    self._config.nonlinear_constraints.lower_bounds.shape,
                    dtype=np.float64,
                )
                constraints.fill(np.nan)
            weighted_objective = np.array(np.nan)
        else:
            objectives = _calculate_estimated_objectives(
                self._config,
                self._function_estimators,
                objectives,
                objective_weights,
                failed_realizations,
            )
            constraints = _calculate_estimated_constraints(
                self._config,
                self._function_estimators,
                constraints,
                constraint_weights,
                failed_realizations,
            )

            weighted_objective = np.array(
                (self._config.objectives.weights * objectives).sum()
            )

        return Functions.create(
            weighted_objective=weighted_objective,
            objectives=objectives,
            constraints=constraints,
        )

    def _compute_gradients(  # noqa: PLR0913
        self,
        variables: NDArray[np.float64],
        mask: NDArray[np.bool_] | None,
        perturbed_variables: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
        perturbed_objectives: NDArray[np.float64],
        perturbed_constraints: NDArray[np.float64] | None,
        objective_weights: NDArray[np.float64] | None,
        constraint_weights: NDArray[np.float64] | None,
        failed_realizations: NDArray[np.bool_],
    ) -> Gradients:
        if mask is not None:
            variables = variables[mask]
        variables = np.repeat(
            variables[np.newaxis, ...], self._config.realizations.weights.size, axis=0
        )

        if mask is not None:
            perturbed_variables = perturbed_variables[..., mask]

        assert perturbed_objectives is not None
        objective_gradients = _calculate_estimated_objective_gradients(
            self._config,
            self._function_estimators,
            variables,
            objectives,
            perturbed_variables,
            perturbed_objectives,
            objective_weights,
            failed_realizations,
        )
        constraint_gradients = _calculate_estimated_constraint_gradients(
            self._config,
            self._function_estimators,
            variables,
            constraints,
            perturbed_variables,
            perturbed_constraints,
            constraint_weights,
            failed_realizations,
        )

        weighted_objective_gradient = np.array(
            (self._config.objectives.weights[:, np.newaxis] * objective_gradients).sum(
                axis=0
            )
        )

        return Gradients.create(
            weighted_objective=self._expand_gradients(
                weighted_objective_gradient, mask
            ),
            objectives=self._expand_gradients(objective_gradients, mask),
            constraints=(
                None
                if constraint_gradients is None
                else self._expand_gradients(constraint_gradients, mask)
            ),
        )

    @staticmethod
    def _expand_gradients(
        gradients: NDArray[np.float64],
        mask: NDArray[np.bool_] | None,
    ) -> NDArray[np.float64]:
        if mask is None:
            return gradients
        shape = gradients.shape[:-1] + (mask.size,)
        result = np.zeros(shape, dtype=np.float64)
        result[..., mask] = gradients
        return result

    def _calculate_filtered_realization_weights(
        self, evaluator_results: _FunctionEvaluatorResults
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        objective_weights: NDArray[np.float64] | None = None
        constraint_weights: NDArray[np.float64] | None = None

        objectives = evaluator_results.objectives
        assert objectives is not None
        constraints = evaluator_results.constraints

        objective_filters = self._config.objectives.realization_filters
        constraint_filters = (
            None
            if self._config.nonlinear_constraints is None
            else self._config.nonlinear_constraints.realization_filters
        )

        for idx, realization_filter in enumerate(self._realization_filters):
            apply_to_objectives = (
                None if objective_filters is None else objective_filters == idx
            )
            apply_to_constraints = (
                None if constraint_filters is None else constraint_filters == idx
            )
            if (apply_to_objectives is None or not np.any(apply_to_objectives)) and (
                apply_to_constraints is None or not np.any(apply_to_constraints)
            ):
                continue

            weights = realization_filter.get_realization_weights(
                objectives, constraints
            )
            if apply_to_objectives is not None:
                if objective_weights is None:
                    objective_weights = np.ones(
                        (
                            self._config.objectives.weights.size,
                            self._config.realizations.weights.size,
                        ),
                    )
                objective_weights[apply_to_objectives, :] = weights
            if constraint_filters is not None and apply_to_constraints is not None:
                assert self._config.nonlinear_constraints is not None
                if constraint_weights is None:
                    constraint_weights = np.ones(
                        (
                            self._config.nonlinear_constraints.lower_bounds.size,
                            self._config.realizations.weights.size,
                        ),
                    )
                constraint_weights[apply_to_constraints, :] = weights
        return objective_weights, constraint_weights

    def _init_realization_filters(
        self, plugin_manager: PluginManager
    ) -> list[RealizationFilter]:
        return [
            plugin_manager.get_plugin(
                "realization_filter", method=filter_config.method
            ).create(self._config, idx)
            for idx, filter_config in enumerate(self._config.realization_filters)
        ]

    def _init_function_estimators(
        self, plugin_manager: PluginManager
    ) -> list[FunctionEstimator]:
        return [
            plugin_manager.get_plugin(
                "function_estimator", method=estimator_config.method
            ).create(self._config, idx)
            for idx, estimator_config in enumerate(self._config.function_estimators)
        ]

    def _init_samplers(
        self, rng: Generator, plugin_manager: PluginManager
    ) -> list[Sampler]:
        samplers: list[Sampler] = []
        for idx, sampler_config in enumerate(self._config.samplers):
            variable_indices = _get_mask(
                idx, self._config.gradient.samplers, self._config.variables.mask
            )
            if variable_indices is None or variable_indices.size:
                plugin = plugin_manager.get_plugin(
                    "sampler", method=sampler_config.method
                )
                samplers.append(plugin.create(self._config, idx, variable_indices, rng))
        return samplers


def _get_mask(
    idx: int, gradient_indices: NDArray[np.intc] | None, mask: NDArray[np.bool_] | None
) -> NDArray[np.bool_] | None:
    if gradient_indices is None:
        return mask
    if mask is None:
        return np.asarray(gradient_indices == idx)
    return np.asarray(mask & (gradient_indices == idx))
