import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.plugins.function_estimator.base import FunctionEstimator


def _calculate_estimated_functions(  # noqa: PLR0913
    config: EnOptConfig,
    function_estimators: list[FunctionEstimator],
    estimator_indices: NDArray[np.intc] | None,
    functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.empty(functions.shape[-1], dtype=np.float64)

    if estimator_indices is None:
        estimator_indices = np.zeros(functions.shape[1], dtype=np.intc)

    for estimator_idx, estimator in enumerate(function_estimators):
        mask = estimator_indices == estimator_idx
        for idx in np.where(mask)[0]:
            weights = (
                config.realizations.weights
                if realization_weights is None
                else realization_weights[idx, ...]
            )
            weights = np.where(failed_realizations, 0.0, weights)
            weights /= weights.sum()
            result[idx] = estimator.calculate_function(functions[..., idx], weights)

    return result


def _calculate_estimated_objectives(
    config: EnOptConfig,
    function_estimators: list[FunctionEstimator],
    functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    return _calculate_estimated_functions(
        config,
        function_estimators,
        config.objectives.function_estimators,
        functions,
        realization_weights,
        failed_realizations,
    )


def _calculate_estimated_constraints(
    config: EnOptConfig,
    function_estimators: list[FunctionEstimator],
    constraints: NDArray[np.float64] | None,
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64] | None:
    if constraints is None:
        return None
    assert config.nonlinear_constraints is not None
    return _calculate_estimated_functions(
        config,
        function_estimators,
        config.nonlinear_constraints.function_estimators,
        constraints,
        realization_weights,
        failed_realizations,
    )
