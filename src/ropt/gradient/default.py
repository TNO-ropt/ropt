"""Default gradient calculator plugin with the stochastic and finite-difference methods."""

from typing import Annotated, ClassVar, Final, override

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ropt._utils import enum_mask
from ropt.config import GradientConfig
from ropt.context import EnOptContext
from ropt.enums import BoundaryType
from ropt.gradient import GradientCalculator, GradientData
from ropt.plugins import MethodSpec
from ropt.sampler import Sampler

_EPSILON: Final = 1e-15
_MIRROR_REPEAT: Final = 3

DEFAULT_GRADIENT_METHODS = {"default", "stochastic", "forward", "backward", "central"}

_STOCHASTIC_FIELDS: Final = (
    "number_of_perturbations",
    "perturbation_min_success",
    "merge_realizations",
)
_PATTERN_TOLERANCE: Final = 1e-6


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


class DefaultGradientCalculator(GradientCalculator):
    """Default gradient calculator with stochastic and finite-difference methods.

    The method is selected via the `method` field of
    [`GradientConfig`][ropt.config.GradientConfig]: `stochastic`, the default,
    or the finite-difference methods `forward`, `backward` and `central`. See
    [Stochastic Gradients](../optimizer_setup/gradients.md) and
    [Finite-Difference Gradients](../optimizer_setup/finite_differences.md)
    for usage.
    """

    methods: ClassVar[MethodSpec] = DEFAULT_GRADIENT_METHODS

    def __init__(self, gradient_config: GradientConfig) -> None:
        """Initialize the gradient calculator.

        Args:
            gradient_config: The gradient configuration.

        Raises:
            ValueError: If the method is not supported, or the options or
                        settings are invalid for the method.
        """
        self._config = gradient_config
        _, _, method = gradient_config.method.lower().rpartition("/")
        if method not in DEFAULT_GRADIENT_METHODS:
            msg = f"Gradient method not supported: {method}"
            raise ValueError(msg)
        self._method = "stochastic" if method == "default" else method
        self._options: StochasticGradientOptions | None = None
        if self._method == "stochastic":
            self._options = StochasticGradientOptions.model_validate(
                gradient_config.options
            )
        else:
            _check_finite_difference_config(gradient_config, self._method)
        self._context: EnOptContext
        self._mask: NDArray[np.bool_]

    @override
    def init(self, context: EnOptContext, *, mask: NDArray[np.bool_]) -> None:
        """Store the context and the mask."""
        self._context = context
        self._mask = mask

    @property
    @override
    def merge_realizations(self) -> bool:
        """The configured `merge_realizations` setting."""
        return self._config.merge_realizations

    @override
    def perturbed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the perturbed points of the selected method.

        Returns:
            The perturbed variables.
        """
        if self._method == "stochastic":
            return _perturb_variables(
                self._context,
                variables,
                self._context.sampler,
                perturbation_count=self._config.number_of_perturbations,
            )
        return _finite_difference_points(
            self._context, variables, self._mask, self._method
        )

    @override
    def failed_realizations(self, data: GradientData) -> NDArray[np.bool_]:
        """Mark realizations with too few successful perturbations as failed.

        Returns:
            For the stochastic method, `True` for each realization with fewer
            than `perturbation_min_success` successful perturbations. For the
            finite-difference methods, `True` for each realization with a
            failed perturbation.
        """
        if self._method != "stochastic":
            # A failed evaluation is NaN in every function, so the first one suffices.
            return np.any(np.isnan(data.perturbed_functions[..., 0]), axis=-1)
        assert self._config.perturbation_min_success is not None
        return _failed_perturbation_realizations(
            data.perturbed_functions, self._config.perturbation_min_success
        )

    @override
    def calculate(
        self, data: GradientData, weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate the gradients with the selected method.

        Returns:
            The gradients, with `NaN` for unavailable gradients.
        """
        if self._method != "stochastic":
            return _estimate_finite_differences(
                data, weights, central=self._method == "central"
            )
        assert self._options is not None
        return _estimate_stochastic(
            data,
            weights,
            merge_realizations=self.merge_realizations,
            svd_threshold=self._options.svd_threshold,
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


def _estimate_stochastic(
    data: GradientData,
    weights: NDArray[np.float64],
    *,
    merge_realizations: bool,
    svd_threshold: float,
) -> NDArray[np.float64]:
    delta_variables = data.perturbed_variables - data.variables
    delta_functions = data.perturbed_functions - np.expand_dims(data.functions, axis=1)
    function_count = delta_functions.shape[-1]
    if merge_realizations:
        return np.stack(
            [
                _estimate_merged_gradient(
                    delta_variables,
                    delta_functions[..., idx],
                    weights[:, idx],
                    svd_threshold=svd_threshold,
                )
                for idx in range(function_count)
            ]
        )
    estimates = [
        _estimate_gradients(
            delta_variables,
            delta_functions[..., idx],
            weights[:, idx],
            svd_threshold=svd_threshold,
        )
        for idx in range(function_count)
    ]
    return np.stack(estimates, axis=1)


def _estimate_gradients(
    delta_variables: NDArray[np.float64],
    delta_functions: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    svd_threshold: float,
) -> NDArray[np.float64]:
    realizations_mask = np.abs(weights) > 0
    realization_count = realizations_mask.size
    gradients: NDArray[np.float64] = np.full(
        (realization_count, delta_variables.shape[-1]), np.nan, dtype=np.float64
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
    return gradients


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
    if not np.any(active_perturbations):
        return np.full(delta_variables.shape[-1], np.nan, dtype=np.float64)
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


def _check_finite_difference_config(
    gradient_config: GradientConfig, method: str
) -> None:
    if gradient_config.options:
        msg = f"The {method} gradient method has no options"
        raise ValueError(msg)
    stochastic_fields = [
        name for name in _STOCHASTIC_FIELDS if name in gradient_config.model_fields_set
    ]
    if stochastic_fields:
        msg = f"The {method} gradient method does not use: " + ", ".join(
            stochastic_fields
        )
        raise ValueError(msg)


def _one_sided_offsets(
    steps: NDArray[np.float64],
    room: NDArray[np.float64],
    other_room: NDArray[np.float64],
) -> NDArray[np.float64]:
    offsets = np.empty_like(steps)
    step_fits_room = steps <= room
    offsets[step_fits_room] = steps[step_fits_room]

    step_fits_other_room = ~step_fits_room & (steps <= other_room)
    offsets[step_fits_other_room] = -steps[step_fits_other_room]

    neither_full_step_fits = ~step_fits_room & ~step_fits_other_room
    use_room = neither_full_step_fits & (room >= other_room)
    offsets[use_room] = room[use_room]

    use_other_room = neither_full_step_fits & (room < other_room)
    offsets[use_other_room] = -other_room[use_other_room]
    return offsets


def _central_difference_offsets(
    steps: NDArray[np.float64],
    room_up: NDArray[np.float64],
    room_down: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    paired = (steps <= room_up) & (steps <= room_down)
    counts = np.ones(steps.size, dtype=np.intp)
    counts[paired] = 2
    starts = np.cumsum(counts) - counts
    targets = np.repeat(np.arange(steps.size, dtype=np.intp), counts)
    offsets = np.empty(targets.size, dtype=np.float64)

    offsets[starts[paired]] = -steps[paired]
    offsets[starts[paired] + 1] = steps[paired]
    single = ~paired
    offsets[starts[single]] = _one_sided_offsets(
        steps[single], room_up[single], room_down[single]
    )
    return targets, offsets


def _finite_difference_points(
    context: EnOptContext,
    variables: NDArray[np.float64],
    mask: NDArray[np.bool_],
    method: str,
) -> NDArray[np.float64]:
    indices = np.flatnonzero(mask)
    values = variables[indices]
    steps = context.variables.perturbation_magnitudes[indices]
    room_up = np.maximum(context.variables.upper_bounds[indices] - values, 0.0)
    room_down = np.maximum(values - context.variables.lower_bounds[indices], 0.0)
    if method == "central":
        positions, offsets = _central_difference_offsets(steps, room_up, room_down)
        targets = indices[positions]
    else:
        targets = indices
        if method == "backward":
            offsets = -_one_sided_offsets(steps, room_down, room_up)
        else:
            offsets = _one_sided_offsets(steps, room_up, room_down)
    points = np.repeat(variables[np.newaxis, :], targets.size, axis=0)
    points[np.arange(targets.size), targets] += offsets
    realization_count = context.realizations.weights.size
    return np.repeat(points[np.newaxis, ...], realization_count, axis=0)


def _estimate_finite_differences(
    data: GradientData, weights: NDArray[np.float64], *, central: bool
) -> NDArray[np.float64]:
    targets = _check_finite_difference_points(data, central=central)
    variable_count = data.variables.size
    counts = np.bincount(targets, minlength=variable_count)
    first_rows = np.cumsum(counts) - counts
    last_rows = first_rows + counts - 1
    # A pair of points gives a central difference, a single point a one-sided
    # difference with the current point.
    pair = (counts > 1)[np.newaxis, :, np.newaxis]
    lower_points = np.where(
        pair, data.perturbed_variables[:, first_rows, :], data.variables
    )
    lower_functions = np.where(
        pair,
        data.perturbed_functions[:, first_rows, :],
        data.functions[:, np.newaxis, :],
    )
    upper_points = data.perturbed_variables[:, last_rows, :]
    upper_functions = data.perturbed_functions[:, last_rows, :]
    steps = np.diagonal(upper_points - lower_points, axis1=1, axis2=2)
    gradients = (upper_functions - lower_functions) / steps[..., np.newaxis]
    gradients = np.swapaxes(gradients, 1, 2)
    estimated = (np.abs(weights) > 0) & ~np.any(np.isnan(gradients), axis=-1)
    return np.where(estimated[..., np.newaxis], gradients, np.nan)


def _check_finite_difference_points(
    data: GradientData, *, central: bool
) -> NDArray[np.intp]:
    variable_count = data.variables.size
    offsets = data.perturbed_variables - data.variables
    row_targets = np.argmax(np.abs(offsets), axis=-1)
    targets: NDArray[np.intp] = row_targets[0]
    steps = np.take_along_axis(offsets, row_targets[..., np.newaxis], axis=-1)
    off_target = targets[:, np.newaxis] != np.arange(variable_count)
    # Supplied initial points are rescaled, which adds rounding errors that
    # grow with the variable offsets.
    tolerance = _PATTERN_TOLERANCE * np.abs(steps)
    deviations = np.where(off_target, np.abs(offsets), 0.0)
    counts = np.bincount(targets, minlength=variable_count)
    allowed_counts = (counts >= 1) & (counts <= (2 if central else 1))
    if (
        np.any(row_targets != targets)
        or np.any(np.diff(targets) < 0)
        or not np.all(allowed_counts)
        or np.any(np.abs(steps) < _EPSILON)
        or np.any(deviations > tolerance)
    ):
        msg = "The perturbed points do not have the finite-difference pattern"
        raise ValueError(msg)
    return targets
