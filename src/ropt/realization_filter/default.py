"""Default realization filter plugin with CVaR methods."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from ropt._utils import apply_direction, zero_failures
from ropt.config import RealizationFilterConfig
from ropt.context import EnOptContext
from ropt.exceptions import TooFewRealizations
from ropt.plugins.realization_filter import RealizationFilterPlugin
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
    with "worst" defined by the constraint type (LE/GE/EQ).
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

    def __init__(self, filter_config: RealizationFilterConfig) -> None:
        """Initialize the realization filter.

        Args:
            filter_config: The realization filter configuration.
        """
        self._filter_config = filter_config
        self._filter_options: CVaRObjectiveOptions | CVaRConstraintOptions

        assert isinstance(self._filter_config, RealizationFilterConfig)
        _, _, self._method = self._filter_config.method.lower().rpartition("/")

    def init(self, context: EnOptContext) -> None:  # ruff: ignore[undocumented-public-method]
        self._context = context

    def get_realization_weights(  # D107  # ruff: ignore[undocumented-public-method]
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        match self._method:
            case "cvar-objective":
                self._filter_options = CVaRObjectiveOptions.model_validate(
                    self._filter_config.options
                )
                weights = self._cvar_objectives(objectives)
            case "cvar-constraint" if constraints is not None:
                self._filter_options = CVaRConstraintOptions.model_validate(
                    self._filter_config.options
                )
                weights = self._cvar_constraint(constraints)
            case _:
                msg = f"Realization filter not supported: {self._method}"
                raise ValueError(msg)

        if not np.any(weights > 0):
            raise TooFewRealizations

        return weights

    def _rank_by(
        self, objectives: NDArray[np.float64], sort: tuple[int, ...]
    ) -> NDArray[np.float64]:
        # The values arrive scaled but not flipped, since direction applies to
        # aggregates and these are per-realization. Ranking is a comparison of
        # what the optimizer is trying to make small, so apply the direction
        # here, per objective and before the weighted sum: one sign cannot
        # stand in for several.
        objective_config = self._context.objectives
        values = zero_failures(objectives[..., sort])
        values = apply_direction(values, objective_config.maximize[sort,])
        if objective_config.weights.size > 1:
            values = np.dot(values, objective_config.weights[sort,])
        return values.flatten()

    def _cvar_objectives(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRObjectiveOptions)
        failed_realizations = np.isnan(objectives[..., 0])
        return _get_cvar_weights_from_percentile(
            -self._rank_by(objectives, self._filter_options.sort),
            failed_realizations,
            self._filter_options.percentile,
        )

    def _cvar_constraint(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRConstraintOptions)
        failed_realizations = np.isnan(constraints[..., 0])
        constraints = zero_failures(constraints[..., self._filter_options.sort])
        assert self._context.nonlinear_constraints is not None
        return _get_cvar_weights_from_percentile(
            -constraints, failed_realizations, self._filter_options.percentile
        )


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


class DefaultRealizationFilterPlugin(RealizationFilterPlugin):
    """Default realization filter plugin class."""

    @classmethod
    def create(cls, filter_config: RealizationFilterConfig) -> DefaultRealizationFilter:
        """Create a DefaultRealizationFilter instance.

        Args:
            filter_config: The realization filter configuration.

        Returns:
            A new `DefaultRealizationFilter`.
        """
        return DefaultRealizationFilter(filter_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in DEFAULT_REALIZATION_FILTER_METHODS
