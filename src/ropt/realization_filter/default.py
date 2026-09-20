"""Default realization filter plugin with CVaR methods."""

from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from ropt._scaling import scale
from ropt._utils import apply_direction, zero_failures
from ropt.config import RealizationFilterConfig
from ropt.exceptions import TooFewRealizations
from ropt.plugins import MethodSpec
from ropt.realization_filter import RealizationFilter

DEFAULT_REALIZATION_FILTER_METHODS = {
    "cvar-objective",
    "cvar-constraint",
}


class _ConfigBaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        frozen=True,
    )


class CVaRObjectiveOptions(_ConfigBaseModel):
    """Options for the `cvar-objective` filter method.

    Assigns CVaR-derived weights to the worst-performing realizations based
    on a weighted sum of objectives.
    See [Realization Filters](../optimizer_setup/realization_filters.md#how-cvar-filters-work)
    for the algorithm.

    Attributes:
        sort:       Objective indices used for the weighted sum.
        percentile: Fraction (0, 1] of worst realizations to include.
    """

    sort: tuple[NonNegativeInt, ...]
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class CVaRConstraintOptions(_ConfigBaseModel):
    """Options for the `cvar-constraint` filter method.

    Assigns CVaR-derived weights based on a single constraint function value,
    ranking realizations by how far each violates that constraint.
    See [Realization Filters](../optimizer_setup/realization_filters.md#how-cvar-filters-work)
    for the algorithm.

    Attributes:
        sort:       Index of the constraint function to use.
        percentile: Fraction (0, 1] of worst realizations to include.
    """

    sort: NonNegativeInt
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class DefaultRealizationFilter(RealizationFilter):
    """Default filter implementation providing CVaR methods.

    The method is selected via the `method` field of
    [`RealizationFilterConfig`][ropt.config.RealizationFilterConfig].
    See [Realization Filters](../optimizer_setup/realization_filters.md) for usage.
    """

    methods: ClassVar[MethodSpec] = DEFAULT_REALIZATION_FILTER_METHODS

    def __init__(self, filter_config: RealizationFilterConfig) -> None:
        """Initialize the realization filter.

        Args:
            filter_config: The realization filter configuration.
        """
        self._filter_config = filter_config
        self._filter_options: CVaRObjectiveOptions | CVaRConstraintOptions

        assert isinstance(self._filter_config, RealizationFilterConfig)
        _, _, self._method = self._filter_config.method.lower().rpartition("/")

    def get_realization_weights(  # D107  # ruff: ignore[undocumented-public-method, too-many-arguments]
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
        *,
        objective_scales: NDArray[np.float64],
        maximize: NDArray[np.bool_],
        objective_weights: NDArray[np.float64],
        constraint_lower_bounds: NDArray[np.float64] | None,
        constraint_upper_bounds: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        match self._method:
            case "cvar-objective":
                self._filter_options = CVaRObjectiveOptions.model_validate(
                    self._filter_config.options
                )
                weights = self._cvar_objectives(
                    objectives, objective_scales, maximize, objective_weights
                )
            case "cvar-constraint" if constraints is not None:
                assert constraint_lower_bounds is not None
                assert constraint_upper_bounds is not None
                self._filter_options = CVaRConstraintOptions.model_validate(
                    self._filter_config.options
                )
                weights = self._cvar_constraint(
                    constraints, constraint_lower_bounds, constraint_upper_bounds
                )
            case _:
                msg = f"Realization filter not supported: {self._method}"
                raise ValueError(msg)

        if not np.any(weights > 0):
            raise TooFewRealizations

        return weights

    def _cvar_objectives(
        self,
        objectives: NDArray[np.float64],
        objective_scales: NDArray[np.float64],
        maximize: NDArray[np.bool_],
        objective_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRObjectiveOptions)
        failed_realizations = np.isnan(objectives[..., 0])
        sort = self._filter_options.sort
        ranked = _rank_by(
            objectives[..., sort],
            objective_scales[sort,],
            maximize[sort,],
            objective_weights[sort,],
        )
        return _get_cvar_weights_from_percentile(
            -ranked,
            failed_realizations,
            self._filter_options.percentile,
        )

    def _cvar_constraint(
        self,
        constraints: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRConstraintOptions)
        sort = self._filter_options.sort
        failed_realizations = np.isnan(constraints[..., sort])
        values = zero_failures(constraints[..., sort])
        # Distance to the violated side: positive when violated, negative slack
        # otherwise, and the absolute difference when the bounds are equal.
        violation = np.maximum(lower_bounds[sort] - values, values - upper_bounds[sort])
        return _get_cvar_weights_from_percentile(
            -violation, failed_realizations, self._filter_options.percentile
        )


def _rank_by(
    objectives: NDArray[np.float64],
    scales: NDArray[np.float64],
    maximize: NDArray[np.bool_],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    # Ranking occurs after scaling and applying the direction for maximization.
    values = zero_failures(objectives)
    values = scale(values, scales)
    values = apply_direction(values, maximize)
    if weights.size > 1:
        values = np.dot(values, weights)
    return values.flatten()


def _get_cvar_weights_from_percentile(
    values: NDArray[np.float64],
    failed_realizations: NDArray[np.bool_],
    percentile: float,
) -> NDArray[np.float64]:
    values = np.where(failed_realizations, np.nan, values)

    indices = np.argsort(values)
    # nan values are sorted to the end, drop them:
    indices = indices[: np.count_nonzero(~failed_realizations)]

    p_max = 1.0 / indices.size
    n_var = int(percentile * indices.size)
    p_var = percentile - n_var * p_max

    weights = np.zeros(values.size)
    weights[indices[:n_var]] = p_max
    if n_var < indices.size:
        weights[indices[n_var]] = p_var
    return weights
