"""Default realization filter plugin with sort and CVaR methods."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from ropt.config import RealizationFilterConfig
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import Abort
from ropt.exit_info import ExitInfo
from ropt.realization_filter import RealizationFilter

DEFAULT_REALIZATION_FILTER_METHODS = {
    "sort-objective",
    "sort-constraint",
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


class SortObjectiveOptions(_ConfigBaseModel):
    """Options for the `sort-objective` filter method.

    Selects realizations by ranking a weighted sum of objectives.
    See [Realization Filters](../usage/realization_filters.md#how-sorting-filters-work)
    for the algorithm.

    Attributes:
        sort:  Objective indices used for the weighted sum.
        first: Starting rank (0-based, inclusive) of selected realizations.
        last:  Ending rank (0-based, inclusive) of selected realizations.
    """

    sort: tuple[NonNegativeInt]
    first: NonNegativeInt
    last: NonNegativeInt


class SortConstraintOptions(_ConfigBaseModel):
    """Options for the `sort-constraint` filter method.

    Selects realizations by ranking a single constraint function value.
    See [Realization Filters](../usage/realization_filters.md#how-sorting-filters-work)
    for the algorithm.

    Attributes:
        sort:  Index of the constraint function to sort by.
        first: Starting rank (0-based, inclusive) of selected realizations.
        last:  Ending rank (0-based, inclusive) of selected realizations.
    """

    sort: NonNegativeInt
    first: NonNegativeInt
    last: NonNegativeInt


class CVaRObjectiveOptions(_ConfigBaseModel):
    """Options for the `cvar-objective` filter method.

    Assigns CVaR-derived weights to the worst-performing realizations based
    on a weighted sum of objectives.
    See [Realization Filters](../usage/realization_filters.md#how-cvar-filters-work)
    for the algorithm.

    Attributes:
        sort:       Objective indices used for the weighted sum.
        percentile: Fraction (0, 1] of worst realizations to include.
    """

    sort: tuple[NonNegativeInt]
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class CVaRConstraintOptions(_ConfigBaseModel):
    """Options for the `cvar-constraint` filter method.

    Assigns CVaR-derived weights based on a single constraint function value,
    with "worst" defined by the constraint type (LE/GE/EQ).
    See [Realization Filters](../usage/realization_filters.md#how-cvar-filters-work)
    for the algorithm.

    Attributes:
        sort:       Index of the constraint function to use.
        percentile: Fraction (0, 1] of worst realizations to include.
    """

    sort: NonNegativeInt
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class DefaultRealizationFilter(RealizationFilter):
    """Default filter implementation providing sort and CVaR methods.

    The method is selected via the `method` field of
    [`RealizationFilterConfig`][ropt.config.RealizationFilterConfig].
    See [Realization Filters](../usage/realization_filters.md) for usage.
    """

    def __init__(  # noqa: D107
        self, filter_config: RealizationFilterConfig
    ) -> None:  # D107
        self._filter_config = filter_config
        self._filter_options: (
            SortObjectiveOptions
            | SortConstraintOptions
            | CVaRObjectiveOptions
            | CVaRConstraintOptions
        )

        assert isinstance(self._filter_config, RealizationFilterConfig)
        _, _, self._method = self._filter_config.method.lower().rpartition("/")

    def init(self, context: EnOptContext) -> None:  # noqa: D102
        self._context = context

    def get_realization_weights(  # D107  # noqa: D102
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        match self._method:
            case "sort-objective":
                self._filter_options = SortObjectiveOptions.model_validate(
                    self._filter_config.options
                )
                self._check_range(self._filter_options)
            case "sort-constraint":
                self._filter_options = SortConstraintOptions.model_validate(
                    self._filter_config.options
                )
                self._check_range(self._filter_options)
            case "cvar-objective":
                self._filter_options = CVaRObjectiveOptions.model_validate(
                    self._filter_config.options
                )
            case "cvar-constraint":
                self._filter_options = CVaRConstraintOptions.model_validate(
                    self._filter_config.options
                )
            case _:
                msg = f"Realization filter not supported: {self._method}"
                raise ValueError(msg)
        weights = self._context.realizations.weights
        match self._method:
            case "sort-objective":
                weights = self._sort_objectives(objectives)
            case "sort-constraint" if constraints is not None:
                weights = self._sort_constraint(constraints)
            case "cvar-objective":
                weights = self._cvar_objectives(objectives)
            case "cvar-constraint" if constraints is not None:
                weights = self._cvar_constraint(constraints)
            case _:
                msg = f"Realization filter not supported: {self._method}"
                raise ValueError(msg)

        if not np.any(weights > 0):
            raise Abort(
                ExitInfo(
                    exit_code=ExitCode.TOO_FEW_REALIZATIONS,
                    message=(
                        "Realization filter produced no realizations with"
                        " non-zero weight"
                    ),
                )
            )

        return weights

    def _check_range(
        self,
        options: SortObjectiveOptions | SortConstraintOptions,
    ) -> None:
        realizations = self._context.realizations.weights.size
        msg = f"Invalid range of realizations: [{options.first}, {options.last}]"
        if options.first < 0 or options.first >= realizations:
            raise ValueError(msg)
        if options.last < 0 or options.last >= realizations:
            raise ValueError(msg)
        if options.last < options.first:
            raise ValueError(msg)

    def _sort_objectives(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, SortObjectiveOptions)
        objective_config = self._context.objectives
        failed_realizations = np.isnan(objectives[..., 0])
        objectives = np.nan_to_num(objectives[..., self._filter_options.sort])
        if objective_config.weights.size > 1:
            objectives = np.dot(
                objectives, objective_config.weights[self._filter_options.sort]
            )
        objectives = objectives.flatten()
        return _sort_and_select(
            objectives,
            self._context.realizations.weights,
            failed_realizations,
            self._filter_options.first,
            self._filter_options.last,
        )

    def _sort_constraint(
        self,
        constraints: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, SortConstraintOptions)
        failed_realizations = np.isnan(constraints[..., 0])
        constraints = np.nan_to_num(constraints[..., self._filter_options.sort])
        return _sort_and_select(
            constraints,
            self._context.realizations.weights,
            failed_realizations,
            self._filter_options.first,
            self._filter_options.last,
        )

    def _cvar_objectives(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRObjectiveOptions)
        objective_config = self._context.objectives
        failed_realizations = np.isnan(objectives[..., 0])
        objectives = np.nan_to_num(objectives[..., self._filter_options.sort])
        if objective_config.weights.size > 1:
            objectives = np.dot(
                objectives, objective_config.weights[self._filter_options.sort]
            )
        objectives = -objectives.flatten()
        return _get_cvar_weights_from_percentile(
            objectives, failed_realizations, self._filter_options.percentile
        )

    def _cvar_constraint(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRConstraintOptions)
        failed_realizations = np.isnan(constraints[..., 0])
        constraints = np.nan_to_num(constraints[..., self._filter_options.sort])
        assert self._context.nonlinear_constraints is not None
        return _get_cvar_weights_from_percentile(
            -constraints, failed_realizations, self._filter_options.percentile
        )


def _sort_and_select(
    values: NDArray[np.float64],
    configured_weights: NDArray[np.float64],
    failed_realizations: NDArray[np.bool_],
    first: int,
    last: int,
) -> NDArray[np.float64]:
    values = np.where(failed_realizations, np.nan, values)
    indices = np.argsort(values)
    # nan values are sorted to the end, drop them:
    indices = indices[: np.count_nonzero(~failed_realizations)]
    indices = indices[first : last + 1]
    weights = np.zeros(configured_weights.size)
    weights[indices] = configured_weights[indices]
    return weights


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
