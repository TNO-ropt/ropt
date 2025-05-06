import numpy as np
from numpy.typing import NDArray

from ropt.plugins.function_estimator.base import FunctionEstimator


def _calculate_estimated_functions(
    function_estimators: list[FunctionEstimator],
    estimator_indices: NDArray[np.intc] | None,
    functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64],
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.empty(functions.shape[-1], dtype=np.float64)

    if estimator_indices is None:
        estimator_indices = np.zeros(functions.shape[1], dtype=np.intc)

    realization_weights = np.broadcast_to(
        realization_weights, (functions.shape[1], realization_weights.shape[-1])
    )

    for estimator_idx, estimator in enumerate(function_estimators):
        mask = estimator_indices == estimator_idx
        for idx in np.where(mask)[0]:
            weights = realization_weights[idx, ...]
            weights = np.where(failed_realizations, 0.0, weights)
            weights /= weights.sum()
            result[idx] = estimator.calculate_function(functions[..., idx], weights)

    return result
