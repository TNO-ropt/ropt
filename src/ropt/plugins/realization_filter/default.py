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
    """Configuration settings for the `sort-objective` realization filter.

    This method sorts realizations based on a weighted sum of objective function
    values and assigns weights only to those within a specified rank range.

    How it works:

    1. A weighted sum is calculated for each realization using the objective
       values specified by the `sort` indices and the corresponding weights
       from the main [`EnOptConfig`][ropt.config.enopt.EnOptConfig]. If only
       one objective index is provided in `sort`, no weighting is applied.
    2. Realizations are sorted based on this calculated value (ascending).
    3. Realizations whose rank falls within the range [`first`, `last`]
       (inclusive) are selected.
    4. The original weights (from `EnOptConfig.realizations.weights`) of the
       selected realizations are retained; all other realizations receive a
       weight of zero. Failed realizations (NaN objective values) are effectively
       given the lowest rank and are excluded before selection.

    Attributes:
        sort:  List of objective function indices to use for sorting.
        first: The starting rank (0-based index) of realizations to select after sorting.
        last:  The ending rank (0-based index) of realizations to select after sorting.
    """

    sort: list[NonNegativeInt]
    first: NonNegativeInt
    last: NonNegativeInt


class SortConstraintOptions(_ConfigBaseModel):
    """Configuration settings for the `sort-constraint` realization filter.

    This method sorts realizations based on the value of a single constraint
    function and assigns weights only to those within a specified rank range.

    How it works:

    1. The values of the constraint function specified by the `sort` index are
       retrieved for each realization.
    2. Realizations are sorted based on these constraint values (ascending).
    3. Realizations whose rank falls within the range [`first`, `last`]
       (inclusive) are selected.
    4. The original weights (from `EnOptConfig.realizations.weights`) of the
       selected realizations are retained; all other realizations receive a
       weight of zero. Failed realizations (NaN constraint values) are effectively
       given the lowest rank and are excluded before selection.

    Attributes:
        sort:  The index of the constraint function to use for sorting.
        first: The starting rank (0-based index) of realizations to select after sorting.
        last:  The ending rank (0-based index) of realizations to select after sorting.
    """

    sort: NonNegativeInt
    first: NonNegativeInt
    last: NonNegativeInt


class CVaRObjectiveOptions(_ConfigBaseModel):
    """Configuration settings for the `cvar-objective` realization filter.

    This method calculates realization weights using the Conditional Value-at-Risk
    (CVaR) approach applied to a weighted sum of objective function values. It
    focuses on the "tail" of the distribution representing the worst-performing
    realizations.

    How it works:

    1. A weighted sum is calculated for each realization using the objective
       values specified by the `sort` indices and the corresponding weights
       from the main [`EnOptConfig`][ropt.config.enopt.EnOptConfig]. If only
       one objective index is provided in `sort`, no weighting is applied.
    2. Realizations are conceptually sorted based on this calculated value
       (ascending, assuming minimization).
    3. The method identifies the subset of realizations corresponding to the
       `percentile` worst outcomes (i.e., the highest weighted objective values).
    4. Weights are assigned to these worst-performing realizations based on the
       CVaR calculation. If the `percentile` boundary falls between two
       realizations, interpolation is used to assign partial weights. All other
       realizations receive a weight of zero.
    5. Failed realizations (NaN objective values) are effectively excluded from
       the CVaR calculation.

    Attributes:
        sort:       List of objective function indices to use for the weighted sum.
        percentile: The CVaR percentile (0.0 to 1.0) defining the portion of
                    worst realizations to consider. Defaults to 0.5.
    """

    sort: list[NonNegativeInt]
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class CVaRConstraintOptions(_ConfigBaseModel):
    """Configuration settings for the `cvar-constraint` realization filter.

    This method calculates realization weights using the Conditional Value-at-Risk
    (CVaR) approach applied to the values of a single constraint function. It
    focuses on the "tail" of the distribution representing the realizations that
    most severely violate or are furthest from satisfying the constraint.

    How it works:

    1. The values of the constraint function specified by the `sort` index are
       retrieved for each realization. These values typically represent the
       constraint function evaluated minus its right-hand-side value (e.g.,
       `g(x) - rhs`).
    2. Realizations are conceptually sorted based on how "badly" they perform
       with respect to the constraint type:
        - **LE (`<=`) constraints:** Realizations with the *largest* positive
          values (most violated) are considered the worst.
        - **GE (`>=`) constraints:** Realizations with the *smallest* negative
          values (most violated) are considered the worst.
        - **EQ (`==`) constraints:** Realizations with the *largest absolute*
          values (furthest from zero) are considered the worst.
    3. The method identifies the subset of realizations corresponding to the
       `percentile` worst outcomes based on the sorting defined above.
    4. Weights are assigned to these worst-performing realizations based on the
       CVaR calculation. If the `percentile` boundary falls between two
       realizations, interpolation is used to assign partial weights. All other
       realizations receive a weight of zero.
    5. Failed realizations (NaN constraint values) are effectively excluded from
       the CVaR calculation.

    Attributes:
        sort:       The index of the constraint function to use.
        percentile: The CVaR percentile (0.0 to 1.0) defining the portion of
                    worst realizations to consider. Defaults to 0.5.
    """

    sort: NonNegativeInt
    percentile: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class DefaultRealizationFilter(RealizationFilter):
    """The default implementation for realization filtering strategies.

    This class provides several methods for calculating realization weights based
    on objective or constraint values. The specific method and its parameters
    are configured via the
    [`RealizationFilterConfig`][ropt.config.enopt.RealizationFilterConfig]
    in the main [`EnOptConfig`][ropt.config.enopt.EnOptConfig].

    **Supported Methods:**

    - `sort-objective`:
        Sorts realizations based on a weighted sum of specified objective
        function values. It then assigns zero weights to realizations outside of
        a defined index range (`first` to `last`) in the sorted list. Requires
        options defined by
        [`SortObjectiveOptions`][ropt.plugins.realization_filter.default.SortObjectiveOptions].

    - `sort-constraint`:
        Sorts realizations based on the value of a single specified constraint
        function. It assigns zero weights to realizations outside of a defined
        index range (`first` to `last`) in the sorted list. Requires options
        defined by
        [`SortConstraintOptions`][ropt.plugins.realization_filter.default.SortConstraintOptions].

    - `cvar-objective`:
        Calculates realization weights using the Conditional Value-at-Risk (CVaR)
        method applied to a weighted sum of specified objective function values.
        Weights are assigned based on a specified `percentile` of the worst-performing
        realizations (highest objective values for minimization). Interpolation is
        used if the percentile boundary falls between realizations.
        Requires options defined by
        [`CVaRObjectiveOptions`][ropt.plugins.realization_filter.default.CVaRObjectiveOptions].

    - `cvar-constraint`:
        Calculates realization weights using the CVaR method applied to the
        value of a single specified constraint function. Weights are assigned
        based on a specified `percentile` of the worst-performing realizations
        (definition of "worst" depends on the constraint type: LE, GE, or EQ).
        Interpolation is used if the percentile boundary falls between
        realizations.
        Requires options defined by
        [`CVaRConstraintOptions`][ropt.plugins.realization_filter.default.CVaRConstraintOptions].
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

    @classmethod
    def create(
        cls, enopt_config: EnOptConfig, filter_index: int
    ) -> DefaultRealizationFilter:
        """Initialize the realization filter plugin.

        See the [ropt.plugins.realization_filter.base.RealizationFilterPlugin][]
        abstract base class.

        # noqa
        """
        return DefaultRealizationFilter(enopt_config, filter_index)

    @classmethod
    def is_supported(cls, method: str) -> bool:
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
