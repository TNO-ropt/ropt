"""Default gradient estimator plugin with the stochastic method."""

from typing import Annotated, ClassVar, Final, override

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ropt._utils import enum_mask
from ropt.config import GradientConfig
from ropt.context import EnOptContext
from ropt.enums import BoundaryType
from ropt.gradient import GradientData, GradientEstimator
from ropt.plugins import MethodSpec
from ropt.sampler import Sampler

_EPSILON: Final = 1e-15
_MIRROR_REPEAT: Final = 3

DEFAULT_GRADIENT_METHODS = {"default", "stochastic"}


class StochasticGradientOptions(BaseModel):
    """Options for the `stochastic` gradient method.

    The gradient is the least-squares solution of the linear equations that
    relate the perturbations to the function differences. The solution is
    computed with a truncated singular value decomposition. See
    [Stochastic Gradients](../optimizer_setup/gradients.md#how-the-gradient-is-estimated)
    for the algorithm.

    Attributes:
        svd_threshold: Fraction (0, 1] of the sum of the squared singular
                       values that the retained singular values must reach.
    """

    svd_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.999

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class DefaultGradientEstimator(GradientEstimator):
    """Default gradient estimator providing the `stochastic` method.

    The method is selected via the `method` field of
    [`GradientConfig`][ropt.config.GradientConfig].
    See [Stochastic Gradients](../optimizer_setup/gradients.md) for usage.
    """

    methods: ClassVar[MethodSpec] = DEFAULT_GRADIENT_METHODS

    def __init__(self, gradient_config: GradientConfig) -> None:
        """Initialize the gradient estimator.

        Args:
            gradient_config: The gradient configuration.

        Raises:
            ValueError: If the method is not supported, or the options are
                        invalid.
        """
        self._config = gradient_config
        _, _, method = gradient_config.method.lower().rpartition("/")
        if method not in {"default", "stochastic"}:
            msg = f"Gradient method not supported: {method}"
            raise ValueError(msg)
        self._options = StochasticGradientOptions.model_validate(
            gradient_config.options
        )
        self._context: EnOptContext

    @override
    def init(self, context: EnOptContext, *, mask: NDArray[np.bool_]) -> None:
        """Store the context."""
        self._context = context

    @property
    @override
    def merge_realizations(self) -> bool:
        """The configured `merge_realizations` setting."""
        return self._config.merge_realizations

    @override
    def perturbed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Perturb the variables with the samples of the configured samplers.

        Returns:
            The perturbed variables.
        """
        return _perturb_variables(
            self._context,
            variables,
            self._context.sampler,
            perturbation_count=self._config.number_of_perturbations,
        )

    @override
    def failed_realizations(self, data: GradientData) -> NDArray[np.bool_]:
        """Mark realizations with too few successful perturbations as failed.

        Returns:
            `True` for each realization with fewer than
            `perturbation_min_success` successful perturbations.
        """
        assert self._config.perturbation_min_success is not None
        return _failed_perturbation_realizations(
            data.perturbed_functions, self._config.perturbation_min_success
        )

    @override
    def estimate(
        self, data: GradientData, weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_] | None]:
        """Estimate the gradients from the perturbations by least squares.

        Returns:
            The gradients and the mask of estimated gradients.
        """
        delta_variables = data.perturbed_variables - data.variables
        delta_functions = data.perturbed_functions - np.expand_dims(
            data.functions, axis=1
        )
        function_count = delta_functions.shape[-1]
        threshold = self._options.svd_threshold
        if self.merge_realizations:
            merged = np.stack(
                [
                    _estimate_merged_gradient(
                        delta_variables,
                        delta_functions[..., idx],
                        weights[:, idx],
                        svd_threshold=threshold,
                    )
                    for idx in range(function_count)
                ]
            )
            return merged, None

        estimates = [
            _estimate_gradients(
                delta_variables,
                delta_functions[..., idx],
                weights[:, idx],
                svd_threshold=threshold,
            )
            for idx in range(function_count)
        ]
        return (
            np.stack([estimate[0] for estimate in estimates], axis=1),
            np.stack([estimate[1] for estimate in estimates], axis=1),
        )


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
    matrix: NDArray[np.float64],
    vector: NDArray[np.float64],
    *,
    svd_threshold: float,
) -> NDArray[np.float64]:
    u, sigma, v = np.linalg.svd(matrix, full_matrices=False)
    sigma2 = sigma**2
    select = np.cumsum(sigma2, axis=-1) / np.sum(sigma2, axis=-1) < svd_threshold
    select[np.argmin(select, axis=-1)] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_inv = np.where(select & (sigma > _EPSILON), 1.0 / sigma, 0.0)
    result: NDArray[np.float64] = np.matmul(v.T, sigma_inv * np.matmul(u.T, vector))
    return result


def _perturb_variables(
    context: EnOptContext,
    variables: NDArray[np.float64],
    samplers: tuple[Sampler, ...],
    *,
    perturbation_count: int,
) -> NDArray[np.float64]:
    # The results should be independent of the order of the samplers,
    # reordering would affect the random numbers they are based on. We
    # obtain a consistent order by running multiple samplers in the order
    # that they appear in the config.variables.samplers array:
    indices = list(
        dict.fromkeys(
            index for index in context.variables.samplers if index is not None
        )
    )
    realization_count = context.realizations.weights.size
    samples = samplers[indices[0]].generate_samples(
        realization_count=realization_count, perturbation_count=perturbation_count
    )
    for index in indices[1:]:
        samples += samplers[index].generate_samples(
            realization_count=realization_count, perturbation_count=perturbation_count
        )
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
    *,
    svd_threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    realizations_mask = np.abs(weights) > 0
    realization_count = realizations_mask.size
    gradients: NDArray[np.float64] = np.zeros(
        (realization_count, delta_variables.shape[-1]), dtype=np.float64
    )
    all_successes = np.logical_not(np.isnan(delta_functions))
    estimated_realizations = np.logical_and(
        realizations_mask, np.any(all_successes, axis=1)
    )
    for idx in range(realization_count):
        success = all_successes[idx]
        if estimated_realizations[idx]:
            ensemble_matrix = delta_variables[idx, success, :]
            gradients[idx, :] = _invert_linear_equations(
                ensemble_matrix,
                delta_functions[idx, success],
                svd_threshold=svd_threshold,
            )
    return gradients, estimated_realizations


def _estimate_merged_gradient(
    delta_variables: NDArray[np.float64],
    delta_functions: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    svd_threshold: float,
) -> NDArray[np.float64]:
    delta_functions = delta_functions.copy() * np.expand_dims(weights, axis=-1)
    realizations_mask = np.abs(weights) > 0
    active_perturbations = np.repeat(realizations_mask, delta_variables.shape[1])
    delta_variables = delta_variables.reshape(-1, delta_variables.shape[-1])
    delta_functions = delta_functions.flatten()
    active_perturbations &= np.logical_not(np.isnan(delta_functions))
    return _invert_linear_equations(
        delta_variables[active_perturbations, :],
        delta_functions[active_perturbations],
        svd_threshold=svd_threshold,
    )


def _failed_perturbation_realizations(
    perturbed_functions: NDArray[np.float64], perturbation_min_success: int
) -> NDArray[np.bool_]:
    # A failed evaluation is NaN in every function, so the first one suffices.
    failed_perturbations = np.isnan(perturbed_functions[..., 0])
    success_count = np.count_nonzero(~failed_perturbations, axis=-1)
    return np.array(success_count < perturbation_min_success)
