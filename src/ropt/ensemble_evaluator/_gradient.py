import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.enums import BoundaryType
from ropt.plugins.function_estimator.base import FunctionEstimator
from ropt.plugins.sampler.base import Sampler

SVD_TOLERANCE = 0.999
MIRROR_REPEAT = 3


def _apply_bounds(
    variables: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    truncation_types: NDArray[np.ubyte],
) -> NDArray[np.float64]:
    def mirror(
        variables: NDArray[np.float64],
        mask: NDArray[np.bool_],
        condition: NDArray[np.bool_],
        bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.where(
            np.logical_and(mask, condition), 2 * bounds - variables, variables
        )

    # Repeat the mirroring a few times, handling mirrored values that still
    # violate the bounds. If that is not sufficient, clip the values.
    mask1 = np.logical_and(
        truncation_types == BoundaryType.MIRROR_BOTH, variables < lower_bounds
    )
    mask2 = np.logical_and(
        truncation_types == BoundaryType.MIRROR_BOTH, variables > upper_bounds
    )
    for _ in range(MIRROR_REPEAT):
        variables = mirror(variables, mask1, variables < lower_bounds, lower_bounds)
        variables = mirror(variables, mask1, variables > upper_bounds, upper_bounds)
    for _ in range(MIRROR_REPEAT):
        variables = mirror(variables, mask2, variables > upper_bounds, upper_bounds)
        variables = mirror(variables, mask2, variables < lower_bounds, lower_bounds)

    # Finally, fall back to clipping.
    return np.clip(variables, lower_bounds, upper_bounds)


def _invert_linear_equations(
    matrix: NDArray[np.float64], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    u, sigma, v = np.linalg.svd(matrix)
    u = u[:, : sigma.size]
    v = v[: sigma.size, :]
    sigma2 = sigma**2
    select = np.cumsum(sigma2) / np.sum(sigma2) < SVD_TOLERANCE
    select[np.argmin(select)] = True  # Add the element that passes the tolerance
    sigma_inv = np.diag(
        np.divide(1.0, sigma, out=np.zeros_like(sigma), where=(sigma > 0) & select)
    )
    result: NDArray[np.float64] = v.T.dot(sigma_inv).dot(u.T).dot(vector)
    return result


def _perturb_variables(
    config: EnOptConfig,
    variables: NDArray[np.float64],
    samplers: list[Sampler],
) -> NDArray[np.float64]:
    if config.gradient.samplers is None:
        samples = samplers[0].generate_samples()
    else:
        # The results should be independent of the order of the samplers,
        # reordering would affect the random numbers they are based on. We
        # obtain a consistent order by running multiple samplers in the order
        # that they appear in the config.gradient.samplers array:
        unique, indices = np.unique(
            np.compress(
                config.gradient.samplers >= 0,
                config.gradient.samplers,
            ),
            return_index=True,
        )
        sampler_indices = unique[np.argsort(indices)]
        samples = samplers[sampler_indices[0]].generate_samples()
        for sampler_idx in sampler_indices[1:]:
            samples += samplers[sampler_idx].generate_samples()
    return _apply_bounds(
        variables + config.gradient.perturbation_magnitudes * samples,
        config.variables.lower_bounds,
        config.variables.upper_bounds,
        config.gradient.boundary_types,
    )


def _estimate_gradients(
    delta_variables: NDArray[np.float64],
    delta_functions: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    active_realizations = np.abs(weights) > 0
    realization_count = active_realizations.size
    gradients: NDArray[np.float64] = np.zeros(
        (delta_variables.shape[-1], realization_count), dtype=np.float64
    )
    all_successes = np.logical_not(np.isnan(delta_functions))
    for idx in range(realization_count):
        success = all_successes[idx]
        if active_realizations[idx] and np.any(success):
            ensemble_matrix = delta_variables[idx, success, :]
            gradients[:, idx] = _invert_linear_equations(
                ensemble_matrix, delta_functions[idx, success]
            )
    return gradients


def _estimate_merged_gradient(
    delta_variables: NDArray[np.float64],
    delta_functions: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    delta_functions = delta_functions * np.expand_dims(weights, axis=-1)
    active_realizations = np.abs(weights) > 0
    active_perturbations = np.repeat(active_realizations, delta_variables.shape[1])
    delta_variables = delta_variables.reshape(-1, delta_variables.shape[-1])
    delta_functions = delta_functions.flatten()
    active_perturbations &= np.logical_not(np.isnan(delta_functions))
    return _invert_linear_equations(
        delta_variables[active_perturbations, :], delta_functions[active_perturbations]
    )


def _calculate_gradient(  # noqa: PLR0913
    functions: NDArray[np.float64],
    delta_variables: NDArray[np.float64],
    delta_functions: NDArray[np.float64],
    failed_realizations: NDArray[np.bool_],
    weights: NDArray[np.float64],
    estimator: FunctionEstimator,
    *,
    merge_realizations: bool,
) -> NDArray[np.float64]:
    weights = np.where(failed_realizations, 0.0, weights)
    weights /= weights.sum()
    if merge_realizations:
        gradients = _estimate_merged_gradient(delta_variables, delta_functions, weights)
    else:
        gradients = _estimate_gradients(delta_variables, delta_functions, weights)
    return estimator.calculate_gradient(functions, gradients, weights)


def _calculate_estimated_gradients(  # noqa: PLR0913
    config: EnOptConfig,
    function_estimators: list[FunctionEstimator],
    estimator_indices: NDArray[np.intc] | None,
    variables: NDArray[np.float64],
    functions: NDArray[np.float64],
    perturbed_variables: NDArray[np.float64],
    perturbed_functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    gradients = np.zeros((functions.shape[-1], variables.shape[-1]), dtype=np.float64)
    delta_variables = perturbed_variables - np.expand_dims(variables, axis=1)
    delta_functions = perturbed_functions - np.expand_dims(functions, axis=1)

    if estimator_indices is None:
        estimator_indices = np.zeros(functions.shape[1], dtype=np.intc)

    for estimator_idx, estimator in enumerate(function_estimators):
        mask = estimator_indices == estimator_idx
        for idx in np.where(mask)[0]:
            gradients[idx, ...] = _calculate_gradient(
                functions[..., idx],
                delta_variables,
                delta_functions[..., idx],
                failed_realizations,
                (
                    config.realizations.weights
                    if realization_weights is None
                    else realization_weights[idx, ...]
                ),
                estimator,
                merge_realizations=config.gradient.merge_realizations,
            )

    return gradients


def _calculate_estimated_objective_gradients(  # noqa: PLR0913
    config: EnOptConfig,
    function_estimators: list[FunctionEstimator],
    variables: NDArray[np.float64],
    functions: NDArray[np.float64],
    perturbed_variables: NDArray[np.float64],
    perturbed_functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    return _calculate_estimated_gradients(
        config,
        function_estimators,
        config.objectives.function_estimators,
        variables,
        functions,
        perturbed_variables,
        perturbed_functions,
        realization_weights,
        failed_realizations,
    )


def _calculate_estimated_constraint_gradients(  # noqa: PLR0913
    config: EnOptConfig,
    function_estimators: list[FunctionEstimator],
    variables: NDArray[np.float64],
    constraints: NDArray[np.float64] | None,
    perturbed_variables: NDArray[np.float64],
    perturbed_constaints: NDArray[np.float64] | None,
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64] | None:
    if constraints is None:
        return None
    assert perturbed_constaints is not None
    assert config.nonlinear_constraints is not None
    return _calculate_estimated_gradients(
        config,
        function_estimators,
        config.nonlinear_constraints.function_estimators,
        variables,
        constraints,
        perturbed_variables,
        perturbed_constaints,
        realization_weights,
        failed_realizations,
    )
