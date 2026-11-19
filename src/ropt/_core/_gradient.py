from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ropt._utils import enum_mask
from ropt.config import GradientConfig
from ropt.context import EnOptContext
from ropt.enums import BoundaryType
from ropt.sampler import Sampler

_SVD_TOLERANCE = 0.999
_EPSILON = 1e-15
_MIRROR_REPEAT = 3


def _apply_bounds(
    variables: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    truncation_types: tuple[BoundaryType, ...],
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
    mirrored = enum_mask(truncation_types, BoundaryType.MIRROR)
    mask1 = np.logical_and(mirrored, variables < lower_bounds)
    mask2 = np.logical_and(mirrored, variables > upper_bounds)
    for _ in range(_MIRROR_REPEAT):
        variables = mirror(variables, mask1, variables < lower_bounds, lower_bounds)
        variables = mirror(variables, mask1, variables > upper_bounds, upper_bounds)
    for _ in range(_MIRROR_REPEAT):
        variables = mirror(variables, mask2, variables > upper_bounds, upper_bounds)
        variables = mirror(variables, mask2, variables < lower_bounds, lower_bounds)

    # Finally, fall back to clipping, except where no correction was asked for.
    return np.where(
        enum_mask(truncation_types, BoundaryType.NONE),
        variables,
        np.clip(variables, lower_bounds, upper_bounds),
    )


def _invert_linear_equations(
    matrix: NDArray[np.float64], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    u, sigma, v = np.linalg.svd(matrix, full_matrices=False)
    sigma2 = sigma**2
    select = np.cumsum(sigma2, axis=-1) / np.sum(sigma2, axis=-1) < _SVD_TOLERANCE
    select[np.argmin(select, axis=-1)] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_inv = np.where(select & (sigma > _EPSILON), 1.0 / sigma, 0.0)
    result: NDArray[np.float64] = np.matmul(v.T, sigma_inv * np.matmul(u.T, vector))
    return result


def _perturb_variables(
    context: EnOptContext,
    variables: NDArray[np.float64],
    samplers: dict[str, Sampler],
) -> NDArray[np.float64]:
    # The results should be independent of the order of the samplers,
    # reordering would affect the random numbers they are based on. We
    # obtain a consistent order by running multiple samplers in the order
    # that they appear in the config.variables.samplers array:
    keys = list(
        dict.fromkeys(key for key in context.variables.samplers if key is not None)
    )
    samples = samplers[keys[0]].generate_samples()
    for key in keys[1:]:
        samples += samplers[key].generate_samples()
    return _apply_bounds(
        variables + context.variables.perturbation_magnitudes * samples,
        context.variables.lower_bounds,
        context.variables.upper_bounds,
        context.variables.boundary_types,
    )


def _estimate_gradients(
    delta_variables: NDArray[np.float64],
    delta_functions: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    realizations_mask = np.abs(weights) > 0
    realization_count = realizations_mask.size
    gradients: NDArray[np.float64] = np.zeros(
        (delta_variables.shape[-1], realization_count), dtype=np.float64
    )
    all_successes = np.logical_not(np.isnan(delta_functions))
    for idx in range(realization_count):
        success = all_successes[idx]
        if realizations_mask[idx] and np.any(success):
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
    delta_functions = delta_functions.copy() * np.expand_dims(weights, axis=-1)
    realizations_mask = np.abs(weights) > 0
    active_perturbations = np.repeat(realizations_mask, delta_variables.shape[1])
    delta_variables = delta_variables.reshape(-1, delta_variables.shape[-1])
    delta_functions = delta_functions.flatten()
    active_perturbations &= np.logical_not(np.isnan(delta_functions))
    return _invert_linear_equations(
        delta_variables[active_perturbations, :], delta_functions[active_perturbations]
    )


def _get_failed_gradient_realizations(
    functions: NDArray[np.float64],
    perturbed_functions: NDArray[np.float64],
    perturbation_min_success: int,
) -> NDArray[np.bool_]:
    # A failed evaluation is NaN in every function, so the first one suffices.
    failed_realizations = np.isnan(functions[..., 0])
    failed_perturbations = np.isnan(perturbed_functions[..., 0])
    success_count = np.count_nonzero(~failed_perturbations, axis=-1)
    failed_realizations |= success_count < perturbation_min_success
    return np.array(failed_realizations)


@dataclass(frozen=True, slots=True)
class _GradientData:
    # Free variables in the optimizer's scaled domain, and the evaluated
    # functions as returned: the objectives, followed by the constraints.
    variables: NDArray[np.float64]
    functions: NDArray[np.float64]
    perturbed_variables: NDArray[np.float64]
    perturbed_functions: NDArray[np.float64]


class _StochasticGradient:
    def __init__(self, gradient_config: GradientConfig) -> None:
        self._config = gradient_config
        self._context: EnOptContext

    def init(self, context: EnOptContext) -> None:
        self._context = context

    @property
    def perturbation_count(self) -> int:
        return self._config.number_of_perturbations

    @property
    def merges_realizations(self) -> bool:
        return self._config.merge_realizations

    def perturbed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _perturb_variables(self._context, variables, self._context.samplers)

    def failed_realizations(self, data: _GradientData) -> NDArray[np.bool_]:
        assert self._config.perturbation_min_success is not None
        return _get_failed_gradient_realizations(
            data.functions,
            data.perturbed_functions,
            self._config.perturbation_min_success,
        )

    def estimate(
        self, data: _GradientData, weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        delta_variables = data.perturbed_variables - data.variables
        delta_functions = data.perturbed_functions - np.expand_dims(
            data.functions, axis=1
        )
        regression = (
            _estimate_merged_gradient
            if self.merges_realizations
            else _estimate_gradients
        )
        return np.stack(
            [
                regression(delta_variables, delta_functions[..., idx], weights[idx])
                for idx in range(delta_functions.shape[-1])
            ]
        )
