from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.plugins.function_transform.protocol import FunctionTransformProtocol


def _calculate_transformed_functions(  # noqa: PLR0913
    config: EnOptConfig,
    function_transforms: List[FunctionTransformProtocol],
    functions: NDArray[np.float64],
    realization_weights: Optional[NDArray[np.float64]],
    failed_realizations: NDArray[np.bool_],
    *,
    constraints: bool = False,
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.empty(functions.shape[-1], dtype=np.float64)

    if constraints:
        assert config.nonlinear_constraints is not None
        transform_indices = config.nonlinear_constraints.function_transforms
    else:
        transform_indices = config.objective_functions.function_transforms

    if transform_indices is None:
        transform_indices = np.zeros(functions.shape[1], dtype=np.intc)

    for transform_idx, transform in enumerate(function_transforms):
        _add_transformed_functions(
            config,
            transform,
            functions,
            realization_weights,
            failed_realizations,
            transform_indices == transform_idx,
            result,
        )

    return result


def _add_transformed_functions(  # noqa: PLR0913
    config: EnOptConfig,
    transform: FunctionTransformProtocol,
    functions: NDArray[np.float64],
    realization_weights: Optional[NDArray[np.float64]],
    failed_realizations: NDArray[np.bool_],
    mask: NDArray[np.bool_],
    result: NDArray[np.float64],
) -> None:
    for idx in np.where(mask)[0]:
        weights = (
            config.realizations.weights
            if realization_weights is None
            else realization_weights[idx, ...]
        )
        weights = np.where(failed_realizations, 0.0, weights)
        weights /= weights.sum()
        result[idx] = transform.calculate_function(functions[..., idx], weights)


def _calculate_weighted_function(
    functions: NDArray[np.float64],
    weights: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.array((weights * functions / scales).sum())
