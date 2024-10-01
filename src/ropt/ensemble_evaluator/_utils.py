from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _compute_auto_scales(
    functions: NDArray[np.float64],
    auto_scale: NDArray[np.bool_],
    weights: NDArray[np.float64],
) -> Optional[NDArray[np.float64]]:
    if np.any(auto_scale):
        weights = np.where(np.isnan(functions[:, 0]), 0.0, weights)
        functions = np.dot(np.nan_to_num(functions).T, weights)
        return np.where(auto_scale, np.fabs(functions[auto_scale]), 1.0)
    return None


def _get_failed_realizations(
    objectives: NDArray[np.float64],
    perturbed_objectives: Optional[NDArray[np.float64]],
    perturbation_min_success: int,
) -> NDArray[np.bool_]:
    failed_realizations = np.isnan(objectives[..., 0])
    if perturbed_objectives is not None:
        failed_pertubations = np.isnan(perturbed_objectives[..., 0])
        success_count = np.count_nonzero(~failed_pertubations, axis=-1)
        failed_realizations |= success_count < perturbation_min_success
    return np.array(failed_realizations)
