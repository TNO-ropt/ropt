import numpy as np
from numpy.typing import NDArray

from ropt.function_estimator import FunctionEstimator


def _calculate_estimated_functions(
    function_estimators: dict[str, FunctionEstimator],
    estimator_keys: tuple[str | None, ...] | None,
    functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64],
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.full(functions.shape[-1], np.nan, dtype=np.float64)

    if estimator_keys is None:
        estimator_keys = ("0",) * functions.shape[1]

    realization_weights = np.broadcast_to(
        realization_weights, (functions.shape[1], realization_weights.shape[-1])
    )

    for key, estimator in function_estimators.items():
        mask = np.array([item == key for item in estimator_keys])
        for idx in np.where(mask)[0]:
            weights = realization_weights[idx, ...]
            weights = np.where(failed_realizations, 0.0, weights)
            weights /= weights.sum()
            result[idx] = estimator.calculate_function(functions[..., idx], weights)

    return result
