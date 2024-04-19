from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig, GradientConfig, VariablesConfig
from ropt.enums import BoundaryType
from ropt.plugins.function_transform.protocol import FunctionTransform
from ropt.plugins.sampler.protocol import Sampler

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
    variables: NDArray[np.float64],
    variables_config: VariablesConfig,
    gradient_config: GradientConfig,
    samplers: List[Sampler],
) -> NDArray[np.float64]:
    if gradient_config.samplers is None:
        samples = samplers[0].generate_samples()
    else:
        # The results should be independent of the order of the samplers,
        # reordering would affect the random numbers they are based on. We
        # obtain a consistent order by running multiple samplers in the order
        # that they appear in the gradient_config.samplers array:
        unique, indices = np.unique(
            np.compress(
                gradient_config.samplers >= 0,
                gradient_config.samplers,
            ),
            return_index=True,
        )
        sampler_indices = unique[np.argsort(indices)]
        samples = samplers[sampler_indices[0]].generate_samples()
        for sampler_idx in sampler_indices[1:]:
            samples += samplers[sampler_idx].generate_samples()
    return _apply_bounds(
        variables + gradient_config.perturbation_magnitudes * samples,
        variables_config.lower_bounds,
        variables_config.upper_bounds,
        gradient_config.boundary_types,
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
    transform: FunctionTransform,
    *,
    merge_realizations: bool,
) -> NDArray[np.float64]:
    weights = np.where(failed_realizations, 0.0, weights)
    weights /= weights.sum()
    if merge_realizations:
        gradients = _estimate_merged_gradient(delta_variables, delta_functions, weights)
    else:
        gradients = _estimate_gradients(delta_variables, delta_functions, weights)
    return transform.calculate_gradient(functions, gradients, weights)


# : disable=too-many-arguments,too-many-locals
def _calculate_transformed_gradients(  # noqa: PLR0913
    config: EnOptConfig,
    function_transforms: List[FunctionTransform],
    variables: NDArray[np.float64],
    functions: NDArray[np.float64],
    perturbed_variables: NDArray[np.float64],
    perturbed_functions: NDArray[np.float64],
    realization_weights: Optional[NDArray[np.float64]],
    failed_realizations: NDArray[np.bool_],
    *,
    constraints: bool = False,
) -> NDArray[np.float64]:
    gradients = np.zeros((functions.shape[-1], variables.shape[-1]), dtype=np.float64)
    delta_variables = perturbed_variables - np.expand_dims(variables, axis=1)
    delta_functions = perturbed_functions - np.expand_dims(functions, axis=1)

    if constraints:
        assert config.nonlinear_constraints is not None
        transform_indices = config.nonlinear_constraints.function_transforms
    else:
        transform_indices = config.objective_functions.function_transforms

    if transform_indices is None:
        transform_indices = np.zeros(functions.shape[1], dtype=np.intc)

    for transform_idx, transform in enumerate(function_transforms):
        _add_transformed_gradients(
            config,
            transform,
            delta_variables,
            functions,
            delta_functions,
            realization_weights,
            failed_realizations,
            transform_indices == transform_idx,
            gradients,
        )

    return gradients


def _add_transformed_gradients(  # noqa: PLR0913
    config: EnOptConfig,
    transform: FunctionTransform,
    delta_variables: NDArray[np.float64],
    functions: NDArray[np.float64],
    delta_functions: NDArray[np.float64],
    realization_weights: Optional[NDArray[np.float64]],
    failed_realizations: NDArray[np.bool_],
    mask: NDArray[np.bool_],
    gradients: NDArray[np.float64],
) -> None:
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
            transform,
            merge_realizations=config.gradient.merge_realizations,
        )


def _calculate_weighted_gradient(
    gradients: NDArray[np.float64],
    weights: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    weights = weights / scales
    return np.array((weights[:, np.newaxis] * gradients).sum(axis=0))
