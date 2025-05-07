from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig

T = TypeVar("T", bound=np.generic)


@dataclass(slots=True)
class EvaluatorContext:
    """Capture additional details for the function evaluator.

    Function evaluator callbacks (see [`Evaluator`][ropt.evaluator.Evaluator])
    primarily receive variable vectors to evaluate objective and constraint
    functions. However, they may also benefit from additional information to
    optimize their calculations. This `EvaluatorContext` object provides that
    supplementary information.

    Specifically, it provides:

    - The configuration object for the current optimization step.
    - The realization index for each variable vector. This can be used to
      determine the correct function from an ensemble to use with each variable
      vector.
    - The perturbation index for each variable vector (if applicable). A value
      less than 0 indicates that the vector is not a perturbation.
    - Boolean matrices (`active_objectives` and `active_constraints`) indicating
      which objective/realization and constraint/realization evaluations are
      required by the optimizer.
    - A boolean vector (`active`) indicating which realizations require
      evaluation.

    The `active_objectives` and `active_constraints` matrices are structured
    such that each column corresponds to a realization, and each row corresponds
    to a objective or constraint. A `True` value signifies that the
    corresponding objective or constraint is essential for the optimizer and
    must be calculated for that realization. Checking if an objective or constraint
    must be calculated for a given control vector involves the following:

    1. Find the realization corresponding to the control in the `realizations`
       property.
    2. Given that realization index, check in `active_objects` and
       `active_constraints` if the objective or constraint should be calculated.

    Note: The `active` property
        In many cases, evaluators may only be able to compute all objectives and
        constraints for a given realization or none at all. In these scenarios,
        the `active` property provides a simplified view, indicating only the
        realizations that need to be evaluated. `active` cannot be set when creating
        the evaluator context, it is calculated from `active_objectives` and
        `active_constraints`. It returns a vector where each entry indicates if
        a given realization is active or not.

    Attributes:
        config:             Configuration of the optimizer.
        realizations:       Realization numbers for each requested evaluation.
        perturbations:      Perturbation numbers for each requested evaluation.
                            A value less than 0 indicates that the vector is
                            not a perturbation.
        active_objectives:  Indicates which objective/realization evaluations are
                            essential for the optimizer.
        active_constraints: Indicates which constraint/realization evaluations
                            are essential for the optimizer.
        active:             Indicates which realizations require evaluation (computed
                            from `active_objectives` and `active_constraints`).
    """

    config: EnOptConfig
    realizations: NDArray[np.intc]
    perturbations: NDArray[np.intc] | None = None
    active_objectives: NDArray[np.bool_] | None = None
    active_constraints: NDArray[np.bool_] | None = None
    active: NDArray[np.bool_] | None = field(init=False)

    def __post_init__(self) -> None:
        active_objectives: NDArray[np.bool_] | None = (
            None
            if self.active_objectives is None
            else np.logical_or.reduce(self.active_objectives, axis=0)
        )
        active_constraints: NDArray[np.bool_] | None = (
            None
            if self.active_constraints is None
            else np.logical_or.reduce(self.active_constraints, axis=0)
        )
        if active_constraints is None:
            self.active = active_objectives
        elif active_objectives is None:
            self.active = active_constraints
        else:
            self.active = np.logical_or(active_objectives, active_constraints)

    def filter_inactive_realizations(self, array: NDArray[T]) -> NDArray[T]:
        """Filter an array based on active realizations.

        This is a utility method, which can be used if only the active property
        is used to exclude realizations that are fully inactive, i.e. where none
        of the objects or constraints are needed.

        This method filters a one- or two-dimensional array by retaining only
        those entries or rows that correspond to active realizations. The
        activity of realizations is determined by the `self.active` boolean
        array (where `True` indicates an active realization). The
        `self.realizations` array maps each input entry to its specific model
        realization index.

        If `self.active` is `None` (indicating that all model realizations are
        to be considered active), no filtering is applied, and the original
        input is returned

        Args:
            array: The array to filter.

        Returns:
            The filtered results.
        """
        if self.active is not None:
            return array[self.active[self.realizations], ...]
        return array

    def insert_inactive_realizations(
        self, array: NDArray[T], *, fill_value: float = 0.0
    ) -> NDArray[T]:
        """Expand an array by inserting fill values for inactive realizations.

        This is a utility method, which can be used if only the active property
        is used to exclude realizations that are fully inactive, i.e. where none
        of the objects or constraints are needed.

        This method takes an array that typically has been processed for active
        realizations (e.g., after being filtered by
        `filter_inactive_realizations`) and expands it to its original
        dimensions by inserting a specified `fill_value` at positions
        corresponding to inactive realizations. If the array is one-dimensional,
        zero entries are inserted, if it is two-dimensional rows of zero values
        are inserted.

        The activity of realizations is determined by `self.active` (a boolean
        array indicating active model realizations) and `self.realizations` (an
        array mapping control vectors to model realization indices). The mask
        `self.active[self.realizations]` identifies which of the original
        control vectors were active.

        If `self.active` is `None` (implying all realizations were considered
        active or no filtering was applied), the input `array` is returned
        unchanged.

        Args:
            array:      The array to expand.
            fill_value: The value to insert for inactive entries.

        Returns:
            An expanded array matching the original number of variables.
        """
        if self.active is not None:
            expanded_array = np.full(
                (self.realizations.shape[0], *array.shape[1:]),
                fill_value=fill_value,
                dtype=array.dtype,
            )
            expanded_array[self.active[self.realizations], ...] = array
            return expanded_array
        return array


@dataclass
class EvaluatorResult:
    """Store the results of a function evaluation.

    This class stores the results of evaluating objective and constraint
    functions for a set of variable vectors.

    The `objectives` and `constraints` are stored as matrices. Each column in
    these matrices corresponds to a specific objective or constraint, and each
    row corresponds to a variable vector.

    When the evaluator is asked to evaluate functions, some variable vectors may
    be marked as inactive. The results for these inactive vectors should be set
    to zero. All active variable vectors should be evaluated. If an evaluation
    fails for any reason, the corresponding values should be set to `numpy.nan`.

    A `batch_id` can be set to identify this specific set of evaluation results.

    The `evaluation_info` dictionary can store additional metadata for each
    evaluation. This information is not used internally by `ropt` and can have
    an arbitrary structure, to be interpreted by the application. This can be
    used, for example, to uniquely identify the results calculated for each
    variable vector, allowing them to be linked back to their corresponding
    input vectors.

    Args:
        objectives:      The calculated objective values.
        constraints:     Optional calculated constraint values.
        batch_id:        Optional batch ID to identify this set of results.
        evaluation_info: Optional info for each evaluation.
    """

    objectives: NDArray[np.float64]
    constraints: NDArray[np.float64] | None = None
    batch_id: int | None = None
    evaluation_info: dict[str, NDArray[Any]] = field(default_factory=dict)


class Evaluator(Protocol):
    """Protocol for evaluator objects or callables.

    The [`EnsembleEvaluator`][ropt.ensemble_evaluator.EnsembleEvaluator] class
    requires a function evaluator callback that conforms to the
    [`Evaluator`][ropt.evaluator.Evaluator] signature. This callback accepts
    one or more variable vectors to evaluate, along with an
    [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] object that provides
    relevant information for the evaluation. It returns an
    [`EvaluatorResult`][ropt.evaluator.EvaluatorResult] object containing the results.
    """

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluatorContext, /
    ) -> EvaluatorResult:
        r"""Evaluate objective and constraint functions for given variables.

        This method defines the signature for the function evaluator callback.
        The evaluator calculates objective and constraint functions for a set of
        variable vectors, potentially for a subset of realizations and
        perturbations.

        Args:
            variables: The matrix of variables to evaluate. Each row represents
                       a variable vector.
            context:   The evaluation context, providing additional information
                       about the evaluation.

        Returns:
            An evaluation results object containing the calculated objective and
                constraint values, along with any additional metadata.

        Tip: Reusing Objective
            When defining multiple objectives, there may be a need to reuse the
            same objective value multiple times. For instance, a total objective
            could consist of the mean of the objectives for each realization,
            plus the standard deviation of the same values. This can be
            implemented by defining two objectives: the first calculated as the
            mean of the realizations, and the second using a function estimator
            to compute the standard deviations. The optimizer is unaware that
            both objectives use the same set of realizations. To prevent
            redundant calculations, the evaluator should compute the results of
            the realizations once and return them for both objectives.
        """
