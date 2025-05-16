from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig

T = TypeVar("T", bound=np.generic)


@dataclass(slots=True)
class EvaluatorContext:
    """Capture additional details for the function evaluator.

    Function evaluators (see [`Evaluator`][ropt.plugins.plan.base.Evaluator])
    primarily receive variable vectors to evaluate objective and constraint
    functions. However, they may also benefit from additional information to
    optimize their calculations. This `EvaluatorContext` object provides that
    supplementary information.

    Specifically, it provides:

    - The configuration object for the current optimization step.
    - A boolean vector (`active`) indicating which realizations require
      evaluation.
    - The realization index for each variable vector. This can be used to
      determine the correct function from an ensemble to use with each variable
      vector.
    - The perturbation index for each variable vector (if applicable). A value
      less than 0 indicates that the vector is not a perturbation.

    Attributes:
        config:             Configuration of the optimizer.
        active:             Indicates which realizations require evaluation.
        realizations:       Realization numbers for each requested evaluation.
        perturbations:      Perturbation numbers for each requested evaluation.
                            A value less than 0 indicates that the vector is
                            not a perturbation.
    """

    config: EnOptConfig
    active: NDArray[np.bool_]
    realizations: NDArray[np.intc]
    perturbations: NDArray[np.intc] | None = None

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
        return array[self.active[self.realizations], ...]

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
        expanded_array = np.full(
            (self.realizations.shape[0], *array.shape[1:]),
            fill_value=fill_value,
            dtype=array.dtype,
        )
        expanded_array[self.active[self.realizations], ...] = array
        return expanded_array


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
