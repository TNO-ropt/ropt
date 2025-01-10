from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
from numpy.random import default_rng

from ropt.results import (
    BoundConstraints,
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Gradients,
    LinearConstraints,
    NonlinearConstraints,
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
from ._function import (
    _calculate_transformed_constraints,
    _calculate_transformed_objectives,
    _calculate_weighted_function,
)
from ._gradient import (
    _calculate_transformed_constraint_gradients,
    _calculate_transformed_objective_gradients,
    _calculate_weighted_gradient,
    _perturb_variables,
)
from ._utils import _compute_auto_scales, _get_failed_realizations

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.evaluator import Evaluator
    from ropt.plugins import PluginManager
    from ropt.plugins.function_transform.base import FunctionTransform
    from ropt.plugins.realization_filter.base import RealizationFilter
    from ropt.plugins.sampler.base import Sampler


class EnsembleEvaluator:
    """A class for constructing functions and gradients from an ensemble of functions.

    This class implements the calculation of functions and gradients from an
    ensemble of functions, based on the settings defined in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] configuration object. It uses
    an [`Evaluator`][ropt.evaluator.Evaluator] callable to evaluate the individual
    functions.
    """

    def __init__(
        self,
        config: EnOptConfig,
        evaluator: Evaluator,
        plan_id: tuple[int, ...],
        result_id_iter: Iterator[int],
        plugin_manager: PluginManager,
    ) -> None:
        """Initialize the ensemble evaluator.

        Args:
            config:         The configuration object.
            evaluator:      The callable for evaluation individual functions.
            plan_id:        A tuple identifying the plan running this evaluator.
            result_id_iter: Iterator for generating consecutive result IDs.
            plugin_manager: A plugin manager to load required plugins.
        """
        self._config = config
        self._evaluator = evaluator
        self._plan_id = plan_id
        self._result_id_iter = result_id_iter
        self._realization_filters = self._init_realization_filters(plugin_manager)
        self._function_transforms = self._init_function_transforms(plugin_manager)
        rng = default_rng(config.gradient.seed)
        self._samplers = self._init_samplers(rng, plugin_manager)
        self._cache_for_gradient: FunctionResults | None = None
        self._objective_auto_scales: NDArray[np.float64] | None = None
        self._constraint_auto_scales: NDArray[np.float64] | None = None

    @property
    def constraint_auto_scales(self) -> NDArray[np.float64] | None:
        """Return optional auto-calculated scales for constraints.

        Returns:
            The calculated scales, or `None` if no scales are available.
        """
        return self._constraint_auto_scales

    def calculate(
        self,
        variables: NDArray[np.float64],
        *,
        compute_functions: bool,
        compute_gradients: bool,
    ) -> tuple[Results, ...]:
        """Evaluate the given variable vectors.

        The `variables` argument may be a single vector of variables or a set
        of variable vectors represented as row-vectors in a matrix. The
        `compute_functions` and `compute_gradients` flags determine which
        results are returned: functions, gradients, or both.

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
                (
                    self._cache_for_gradient.evaluations.variables
                    if self._cache_for_gradient.evaluations.scaled_variables is None
                    else self._cache_for_gradient.evaluations.scaled_variables
                ),
                variables,
                rtol=0.0,
                atol=1e-15,
            )
        ):
            # This assumes that the cached function value is calculated with the
            # same parameters as the gradient is to be calculated. This is true
            # because each optimization step is a single optimizer run and uses
            # its own copy of an `EnsembleFunctionEvaluation` object.
            return self._calculate_gradients(variables, self._config.variables.indices)

        # A function + gradient, or a gradient without cached function:
        self._cache_for_gradient = None
        return self._calculate_both(variables, self._config.variables.indices)

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
        # Autoscaling is done by finding the weighted mean of the realizations:
        if self._objective_auto_scales is None:
            self._objective_auto_scales = _compute_auto_scales(
                f_eval_results.objectives,
                self._config.objectives.auto_scale,
                self._config.realizations.weights,
            )
        if (
            f_eval_results.constraints is not None
            and self._constraint_auto_scales is None
        ):
            assert self._config.nonlinear_constraints is not None
            self._constraint_auto_scales = _compute_auto_scales(
                f_eval_results.constraints,
                self._config.nonlinear_constraints.auto_scale,
                self._config.realizations.weights,
            )

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
            config=self._config,
            objective_auto_scales=self._objective_auto_scales,
            constraint_auto_scales=self._constraint_auto_scales,
            variables=variables,
            objectives=f_eval_results.objectives,
            constraints=f_eval_results.constraints,
            evaluation_ids=f_eval_results.evaluation_ids,
        )

        return FunctionResults(
            plan_id=self._plan_id,
            result_id=next(self._result_id_iter),
            batch_id=f_eval_results.batch_id,
            metadata={},
            evaluations=evaluations,
            realizations=Realizations(
                failed_realizations=failed_realizations,
                objective_weights=objective_weights,
                constraint_weights=constraint_weights,
            ),
            functions=functions,
            bound_constraints=BoundConstraints.create(self._config, evaluations),
            linear_constraints=LinearConstraints.create(self._config, evaluations),
            nonlinear_constraints=NonlinearConstraints.create(
                self._config, functions, self._constraint_auto_scales
            ),
        )

    def _calculate_gradients(
        self,
        variables: NDArray[np.float64],
        variable_indices: NDArray[np.intc] | None,
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
                variable_indices,
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
                plan_id=self._plan_id,
                result_id=next(self._result_id_iter),
                batch_id=g_eval_results.batch_id,
                metadata={},
                evaluations=GradientEvaluations.create(
                    config=self._config,
                    variables=variables,
                    perturbed_variables=perturbed_variables,
                    perturbed_objectives=g_eval_results.perturbed_objectives,
                    perturbed_constraints=g_eval_results.perturbed_constraints,
                    objective_auto_scales=self._objective_auto_scales,
                    constraint_auto_scales=self._constraint_auto_scales,
                    perturbed_evaluation_ids=g_eval_results.perturbed_evaluation_ids,
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
        variable_indices: NDArray[np.intc] | None,
    ) -> tuple[FunctionResults, GradientResults]:
        perturbed_variables = _perturb_variables(
            self._config, variables, self._samplers
        )
        active_objectives, active_constraints = _get_active_realizations(self._config)
        f_eval_results, g_eval_results = _get_function_and_gradient_results(
            self._config,
            self._evaluator,
            variables,
            perturbed_variables,
            active_objectives,
            active_constraints,
        )

        # Autoscaling is done by finding the weighted mean of the realizations:
        if self._objective_auto_scales is None:
            self._objective_auto_scales = _compute_auto_scales(
                f_eval_results.objectives,
                self._config.objectives.auto_scale,
                self._config.realizations.weights,
            )
        if (
            f_eval_results.constraints is not None
            and self._constraint_auto_scales is None
        ):
            assert self._config.nonlinear_constraints is not None
            self._constraint_auto_scales = _compute_auto_scales(
                f_eval_results.constraints,
                self._config.nonlinear_constraints.auto_scale,
                self._config.realizations.weights,
            )

        evaluations = FunctionEvaluations.create(
            config=self._config,
            objective_auto_scales=self._objective_auto_scales,
            constraint_auto_scales=self._constraint_auto_scales,
            variables=variables,
            objectives=f_eval_results.objectives,
            constraints=f_eval_results.constraints,
            evaluation_ids=f_eval_results.evaluation_ids,
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
            plan_id=self._plan_id,
            result_id=next(self._result_id_iter),
            batch_id=f_eval_results.batch_id,
            metadata={},
            evaluations=evaluations,
            realizations=Realizations(
                failed_realizations=failed_realizations,
                objective_weights=objective_weights,
                constraint_weights=constraint_weights,
            ),
            functions=functions,
            bound_constraints=BoundConstraints.create(self._config, evaluations),
            linear_constraints=LinearConstraints.create(self._config, evaluations),
            nonlinear_constraints=NonlinearConstraints.create(
                self._config, functions, self._constraint_auto_scales
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
                variable_indices,
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
            plan_id=self._plan_id,
            result_id=next(self._result_id_iter),
            batch_id=g_eval_results.batch_id,
            metadata={},
            evaluations=GradientEvaluations.create(
                config=self._config,
                variables=variables,
                perturbed_variables=perturbed_variables,
                perturbed_objectives=g_eval_results.perturbed_objectives,
                perturbed_constraints=g_eval_results.perturbed_constraints,
                objective_auto_scales=self._objective_auto_scales,
                constraint_auto_scales=self._constraint_auto_scales,
                perturbed_evaluation_ids=g_eval_results.perturbed_evaluation_ids,
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
        # realizations using one or more function transforms:
        if np.all(failed_realizations):
            objectives = np.empty(
                self._config.objectives.weights.size, dtype=np.float64
            )
            objectives.fill(np.nan)
            if constraints is not None:
                assert self._config.nonlinear_constraints is not None
                constraints = np.empty(
                    self._config.nonlinear_constraints.rhs_values, dtype=np.float64
                )
                constraints.fill(np.nan)
            weighted_objective = np.array(np.nan)
        else:
            objectives = _calculate_transformed_objectives(
                self._config,
                self._function_transforms,
                objectives,
                objective_weights,
                failed_realizations,
            )
            constraints = _calculate_transformed_constraints(
                self._config,
                self._function_transforms,
                constraints,
                constraint_weights,
                failed_realizations,
            )

            weighted_objective = _calculate_weighted_function(
                objectives,
                self._config.objectives.weights,
                self._get_objective_scales(self._objective_auto_scales),
            )

        return Functions.create(
            config=self._config,
            weighted_objective=weighted_objective,
            objectives=objectives,
            constraints=constraints,
            objective_auto_scales=self._objective_auto_scales,
            constraint_auto_scales=self._constraint_auto_scales,
        )

    def _compute_gradients(  # noqa: PLR0913
        self,
        variables: NDArray[np.float64],
        variable_indices: NDArray[np.intc] | None,
        perturbed_variables: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
        perturbed_objectives: NDArray[np.float64],
        perturbed_constraints: NDArray[np.float64] | None,
        objective_weights: NDArray[np.float64] | None,
        constraint_weights: NDArray[np.float64] | None,
        failed_realizations: NDArray[np.bool_],
    ) -> Gradients:
        if variable_indices is not None:
            variables = variables[variable_indices]
        variables = np.repeat(
            variables[np.newaxis, ...], self._config.realizations.weights.size, axis=0
        )

        if variable_indices is not None:
            perturbed_variables = perturbed_variables[..., variable_indices]

        assert perturbed_objectives is not None
        objective_gradients = _calculate_transformed_objective_gradients(
            self._config,
            self._function_transforms,
            variables,
            objectives,
            perturbed_variables,
            perturbed_objectives,
            objective_weights,
            failed_realizations,
        )
        constraint_gradients = _calculate_transformed_constraint_gradients(
            self._config,
            self._function_transforms,
            variables,
            constraints,
            perturbed_variables,
            perturbed_constraints,
            constraint_weights,
            failed_realizations,
        )

        weighted_objective_gradient = _calculate_weighted_gradient(
            objective_gradients,
            self._config.objectives.weights,
            self._get_objective_scales(self._objective_auto_scales),
        )

        return Gradients.create(
            config=self._config,
            weighted_objective=self._expand_gradients(
                weighted_objective_gradient, variable_indices
            ),
            objectives=self._expand_gradients(objective_gradients, variable_indices),
            constraints=(
                None
                if constraint_gradients is None
                else self._expand_gradients(constraint_gradients, variable_indices)
            ),
            objective_auto_scales=self._objective_auto_scales,
            constraint_auto_scales=self._constraint_auto_scales,
        )

    def _expand_gradients(
        self,
        gradients: NDArray[np.float64],
        variable_indices: NDArray[np.intc] | None,
    ) -> NDArray[np.float64]:
        if variable_indices is None:
            return gradients
        shape = gradients.shape[:-1] + (self._config.variables.initial_values.size,)
        result = np.zeros(shape, dtype=np.float64)
        result[..., variable_indices] = gradients
        return result

    def _calculate_filtered_realization_weights(
        self, evaluator_results: _FunctionEvaluatorResults
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        objective_weights: NDArray[np.float64] | None = None
        constraint_weights: NDArray[np.float64] | None = None

        objectives = evaluator_results.objectives
        assert objectives is not None
        constraints = evaluator_results.constraints

        objectives = objectives / self._get_objective_scales(
            self._objective_auto_scales
        )

        if constraints is not None:
            assert self._config.nonlinear_constraints is not None
            rhs_values = self._config.nonlinear_constraints.rhs_values
            scales = self._get_constraint_scales(self._constraint_auto_scales)
            constraints = (constraints - rhs_values) / scales

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
                            self._config.nonlinear_constraints.rhs_values.size,
                            self._config.realizations.weights.size,
                        ),
                    )
                constraint_weights[apply_to_constraints, :] = weights
        return objective_weights, constraint_weights

    def _get_objective_scales(
        self, auto_scales: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        if auto_scales is None:
            return self._config.objectives.scales
        return self._config.objectives.scales * auto_scales

    def _get_constraint_scales(
        self, auto_scales: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        assert self._config.nonlinear_constraints is not None
        if auto_scales is None:
            return self._config.nonlinear_constraints.scales
        return self._config.nonlinear_constraints.scales * auto_scales

    def _init_realization_filters(
        self, plugin_manager: PluginManager
    ) -> list[RealizationFilter]:
        return [
            plugin_manager.get_plugin(
                "realization_filter", method=filter_config.method
            ).create(self._config, idx)
            for idx, filter_config in enumerate(self._config.realization_filters)
        ]

    def _init_function_transforms(
        self, plugin_manager: PluginManager
    ) -> list[FunctionTransform]:
        return [
            plugin_manager.get_plugin(
                "function_transform", method=transform_config.method
            ).create(self._config, idx)
            for idx, transform_config in enumerate(self._config.function_transforms)
        ]

    def _init_samplers(
        self, rng: Generator, plugin_manager: PluginManager
    ) -> list[Sampler]:
        samplers: list[Sampler] = []
        for idx, sampler_config in enumerate(self._config.samplers):
            variable_indices = _get_indices(
                idx, self._config.gradient.samplers, self._config.variables.indices
            )
            if variable_indices is None or variable_indices.size:
                plugin = plugin_manager.get_plugin(
                    "sampler", method=sampler_config.method
                )
                samplers.append(plugin.create(self._config, idx, variable_indices, rng))
        return samplers


def _get_indices(
    idx: int,
    gradient_indices: NDArray[np.intc] | None,
    variable_indices: NDArray[np.intc] | None,
) -> NDArray[np.intc] | None:
    if gradient_indices is None:
        return variable_indices
    if variable_indices is None:
        return np.asarray(np.where(gradient_indices == idx)[0], dtype=np.intc)
    indices = np.zeros(gradient_indices.shape, dtype=np.bool_)
    indices[variable_indices] = True
    indices &= gradient_indices == idx
    return np.asarray(np.where(indices)[0], dtype=np.intc)
