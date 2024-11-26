import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.plugins.function_transform.base import FunctionTransform


def _calculate_transformed_functions(  # noqa: PLR0913
    config: EnOptConfig,
    function_transforms: list[FunctionTransform],
    transform_indices: NDArray[np.intc] | None,
    functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.empty(functions.shape[-1], dtype=np.float64)

    if transform_indices is None:
        transform_indices = np.zeros(functions.shape[1], dtype=np.intc)

    for transform_idx, transform in enumerate(function_transforms):
        mask = transform_indices == transform_idx
        for idx in np.where(mask)[0]:
            weights = (
                config.realizations.weights
                if realization_weights is None
                else realization_weights[idx, ...]
            )
            weights = np.where(failed_realizations, 0.0, weights)
            weights /= weights.sum()
            result[idx] = transform.calculate_function(functions[..., idx], weights)

    return result


def _calculate_transformed_objectives(
    config: EnOptConfig,
    function_transforms: list[FunctionTransform],
    functions: NDArray[np.float64],
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64]:
    return _calculate_transformed_functions(
        config,
        function_transforms,
        config.objectives.function_transforms,
        functions,
        realization_weights,
        failed_realizations,
    )


def _calculate_transformed_constraints(
    config: EnOptConfig,
    function_transforms: list[FunctionTransform],
    constraints: NDArray[np.float64] | None,
    realization_weights: NDArray[np.float64] | None,
    failed_realizations: NDArray[np.bool_],
) -> NDArray[np.float64] | None:
    if constraints is None:
        return None
    assert config.nonlinear_constraints is not None
    return _calculate_transformed_functions(
        config,
        function_transforms,
        config.nonlinear_constraints.function_transforms,
        constraints,
        realization_weights,
        failed_realizations,
    )


def _calculate_weighted_function(
    functions: NDArray[np.float64],
    weights: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.array((weights * functions / scales).sum())
