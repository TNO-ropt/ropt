"""This plugin contains realization filters that are installed by default."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from ropt.config.enopt import EnOptConfig
from ropt.enums import OptimizerExitCode
from ropt.exceptions import ConfigError, OptimizationAborted

from .base import RealizationFilter, RealizationFilterPlugin


class _ConfigBaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
    )


class SortObjectiveOptions(_ConfigBaseModel):
    """Configuration settings for the `sort-objective` method.

    The `sort-objective` method sorts realizations according to the value of one
    or multiple objectives, and retains a number of realizations within a given
    index range in the sorted list. If more than one objective index is given, a
    weighted sum of these objectives is used, using the weights given in the
    configuration of the optimizer.

    Attributes:
        sort:  The indices of the objectives to sort.
        first: Index or name of the first realization to use.
        last:  Index of name of the last realization to use.
    """

    sort: list[NonNegativeInt]
    first: NonNegativeInt
    last: NonNegativeInt


class SortConstraintOptions(_ConfigBaseModel):
    """Configuration settings for the `sort-constraint` method.

    The `sort-constraint` method sorts realizations according to the value of a
    constraint, and retains a number of realizations within a given index range
    in the sorted list.

    Attributes:
        sort:  The index of the constraint to sort.
        first: Index or name of the first realization to use.
        last:  Index or name of the last realization to use.
    """

    sort: NonNegativeInt
    first: NonNegativeInt
    last: NonNegativeInt


class CVaRObjectiveOptions(_ConfigBaseModel):
    """Configuration settings for the `cvar-objective` method.

    The `cvar-objective` method finds realizations weights by applying the CVaR
    method to the objective values. If more than one objective index is given, a
    weighted sum of these objectives is used, using the weights given in the
    configuration of the optimizer.

    The percentile argument defines the contribution of the "worst" performing
    realizations in the distribution that is used to calculate the ensemble
    value. "Worst" is defined as those realizations having the highest values in
    case of a minimization and those having the lowest values in case of
    maximizing.

    Attributes:
        sort:       The indices of the objectives to sort.
        percentile: The CVaR percentile.
    """

    sort: list[NonNegativeInt]
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class CVaRConstraintOptions(_ConfigBaseModel):
    """Configuration settings for the `cvar-constraint` method.

    The `cvar-constraint` method finds realizations weights by applying the CVaR
    method to the objective values.

    The percentile argument defines the contribution of the "worst" performing
    realizations in the distribution that is used to calculate the ensemble
    value. The definition of worst depends on the type of the constraints. After
    subtracting the right-hand-side value the following applies:

    - For LE constraints, realizations with the largest values are the worst
    - For GE constraints, realizations with the smallest values are the worst
    - For EQ constraints, realizations with the largest absolute values are the worst

    Attributes:
        sort:       The index of the constraint to sort.
        percentile: The CVaR percentile.
    """

    sort: NonNegativeInt
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class DefaultRealizationFilter(RealizationFilter):
    """The default realization filter plugin class.

    This plugin currently implements four methods:

    `sort-objective`:
    :  Filter realizations by selecting a range of objective values. This filter
       requires additional configuration using an options dict that can be
       parsed into a
       [`SortObjectiveOptions`][ropt.plugins.realization_filter.default.SortObjectiveOptions]
       class. This method sorts realizations according to the weighted sum of
       the values of objective functions specified in the options. It then
       selects the set of realizations from a given index range.

    `sort-constraint`:
    :  Filter realizations by selecting a range of constraint values. This
       filter requires additional configuration using an options dict that can
       be parsed into a
       [`SortConstraintOptions`][ropt.plugins.realization_filter.default.SortConstraintOptions]
       class. This method sorts realizations according to the values of
       constraint functions specified in the options. It then selects the set of
       realizations from a given index range.

    `cvar-objective`:
    :  Filter realizations by selecting a range of objective values. This filter
       requires additional configuration using an options dict that can be
       parsed into a
       [`CVaRObjectiveOptions`][ropt.plugins.realization_filter.default.CVaRObjectiveOptions]
       class. This method sorts realizations according to the weighted sum of
       the values of objective functions specified in the options. It then
       selects a percentile of the realizations, applying interpolation whenever
       the number of selected realizations is not an integer number.

    `cvar-constraint`:
    :  Filter realizations by selecting a range of constraint values. This
       filter requires additional configuration using an options dict that can
       be parsed into a
       [`CVaRConstraintOptions`][ropt.plugins.realization_filter.default.CVaRConstraintOptions]
       class. This method sorts realizations according to the values of
       constraint functions specified in the options. It then selects a
       percentile of the realizations, applying interpolation whenever the
       number of selected realizations is not an integer number.
    """

    def __init__(self, enopt_config: EnOptConfig, filter_index: int) -> None:  # D107
        """Initialize the realization filter plugin.

        See the
        [ropt.plugins.realization_filter.base.RealizationFilter][]
        base class.

        # noqa
        """
        self._filter_config = enopt_config.realization_filters[filter_index]
        self._enopt_config = enopt_config
        self._filter_options: (
            SortObjectiveOptions
            | SortConstraintOptions
            | CVaRObjectiveOptions
            | CVaRConstraintOptions
        )

        _, _, self._method = self._filter_config.method.lower().rpartition("/")
        options = self._filter_config.options
        match self._method:
            case "sort-objective":
                self._filter_options = SortObjectiveOptions.model_validate(options)
                self._check_range(self._filter_options)
            case "sort-constraint":
                self._filter_options = SortConstraintOptions.model_validate(options)
                self._check_range(self._filter_options)
            case "cvar-objective":
                self._filter_options = CVaRObjectiveOptions.model_validate(options)
            case "cvar-constraint":
                self._filter_options = CVaRConstraintOptions.model_validate(options)
            case _:
                msg = f"Realization filter not supported: {self._method}"
                raise ConfigError(msg)

    def get_realization_weights(  # D107
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return the updated weights of the realizations.

        See the
        [ropt.plugins.realization_filter.base.RealizationFilter][]
        abstract base class.

        # noqa
        """
        weights = self._enopt_config.realizations.weights
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
                raise ConfigError(msg)

        if not np.any(weights > 0):
            raise OptimizationAborted(exit_code=OptimizerExitCode.TOO_FEW_REALIZATIONS)

        return weights

    def _check_range(
        self, options: SortObjectiveOptions | SortConstraintOptions
    ) -> None:
        realizations = self._enopt_config.realizations.weights.size
        msg = f"Invalid range of realizations: [{options.first}, {options.last}]"
        if options.first < 0 or options.first >= realizations:
            raise ConfigError(msg)
        if options.last < 0 or options.last >= realizations:
            raise ConfigError(msg)
        if options.last < options.first:
            raise ConfigError(msg)

    def _sort_objectives(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, SortObjectiveOptions)
        objective_config = self._enopt_config.objectives
        failed_realizations = np.isnan(objectives[..., 0])
        objectives = np.nan_to_num(objectives[..., self._filter_options.sort])
        if objective_config.weights.size > 1:
            objectives = np.dot(
                objectives, objective_config.weights[self._filter_options.sort]
            )
        objectives = objectives.flatten()
        return _sort_and_select(
            objectives,
            self._enopt_config.realizations.weights,
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
            self._enopt_config.realizations.weights,
            failed_realizations,
            self._filter_options.first,
            self._filter_options.last,
        )

    def _cvar_objectives(
        self,
        objectives: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRObjectiveOptions)
        objective_config = self._enopt_config.objectives
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

    def _cvar_constraint(
        self,
        constraints: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert isinstance(self._filter_options, CVaRConstraintOptions)
        failed_realizations = np.isnan(constraints[..., 0])
        constraints = np.nan_to_num(constraints[..., self._filter_options.sort])
        assert self._enopt_config.nonlinear_constraints is not None
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


class DefaultRealizationFilterPlugin(RealizationFilterPlugin):
    """Default realization filter plugin class."""

    def create(
        self, enopt_config: EnOptConfig, filter_index: int
    ) -> DefaultRealizationFilter:
        """Initialize the realization filter plugin.

        See the [ropt.plugins.realization_filter.base.RealizationFilterPlugin][]
        abstract base class.

        # noqa
        """
        return DefaultRealizationFilter(enopt_config, filter_index)

    def is_supported(self, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in {
            "sort-objective",
            "sort-constraint",
            "cvar-objective",
            "cvar-constraint",
        }
