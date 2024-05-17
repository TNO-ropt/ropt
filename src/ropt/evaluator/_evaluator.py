from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig


@dataclass
class EvaluatorContext:
    """Capture additional details for the function evaluator.

    Function evaluator callbacks (see [`Evaluator`][ropt.evaluator.Evaluator])
    mainly require variable vectors to evaluate objective and constraint
    functions. However, depending on their implementation, evaluators may
    benefit from additional information. To accommodate this, function
    evaluators receive a `EvaluatorContext` object with the following details:

    - The configuration object for the ongoing optimization step.
    - Indices indicating the realization to which each variable vector belongs.
    - A matrix indicating, for each function and realization, whether it is
      active and needs computation.
    - A matrix indicating, for each constraint and realization, whether it is
      active and requires computation.

    The `active_objectives` and `active_constraints` fields are boolean
    matrices, where each column represents one realization, and each row
    signifies a function or a constraint. Entries marked as `True` are essential
    for the optimizer, while other combinations do not necessitate evaluation.

    In practical scenarios, these matrices may prove overly detailed for
    function evaluators. Typically, evaluators may only be capable of
    calculating all objective and constraint functions for a given realization
    or none at all. In such cases, it suffices to examine the `active` property,
    indicating the realizations requiring evaluation.

    Attributes:
        config:             Configuration of the optimizer.
        realizations:       Realization numbers for each requested evaluation.
        active_objectives:  Signifies which function/realization evaluations are
                            essential for the optimizer.
        active_constraints: Signifies which constraint/realization evaluations are
                            essential for the optimizer.
    """

    config: EnOptConfig
    realizations: NDArray[np.intc]
    active_objectives: Optional[NDArray[np.bool_]] = None
    active_constraints: Optional[NDArray[np.bool_]] = None

    @cached_property
    def active(self) -> Optional[NDArray[np.bool_]]:
        """Return the set of active variable vectors.

        This property is useful for determining the realizations for which the
        objective and constraint functions need to be calculated. The index of
        each entry corresponds to the realization number and indicates whether
        the functions should be calculated.

        Returns:
            A boolean array.
        """
        active_objectives: Optional[NDArray[np.bool_]] = (
            None
            if self.active_objectives is None
            else np.logical_or.reduce(self.active_objectives, axis=0)
        )
        active_constraints: Optional[NDArray[np.bool_]] = (
            None
            if self.active_constraints is None
            else np.logical_or.reduce(self.active_constraints, axis=0)
        )
        if active_constraints is None:
            return active_objectives
        if active_objectives is None:
            return active_constraints
        return np.logical_or(active_objectives, active_constraints)


@dataclass
class EvaluatorResult:
    """Store the results of a function evaluation.

    The objectives and constraint values are stored as a matrix, where the
    columns correspond to the index of the objective or constraint, and the rows
    correspond to the index of the variable vector for which they were
    calculated. Depending on context information passed to the evaluation
    function, not all results may have been calculated, in which case the
    corresponding entries should contain zeros. Entries may also contain
    `numpy.nan` values to signify that a calculation failed.

    Optionally, a batch ID can be returned to identify the batch of
    calculations. This can be useful for tracking or managing evaluations
    performed together.

    Additionally, evaluation IDs are provided as an option. These IDs can be
    used to uniquely identify the results calculated for each variable vector,
    offering a way to link specific evaluations back to their corresponding
    input vectors.

    Attributes:
        objectives:     The calculated objective values.
        constraints:    Optional calculated constraint values.
        batch_id:       Optional batch ID.
        evaluation_ids: Optional ID for each evaluation.
    """

    objectives: NDArray[np.float64]
    constraints: Optional[NDArray[np.float64]] = None
    batch_id: Optional[int] = None
    evaluation_ids: Optional[NDArray[np.intc]] = None


class Evaluator(Protocol):
    """Protocol for evaluator objects or callables.

    The optimizers require a callback that follows this protocol.
    """

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluatorContext, /
    ) -> EvaluatorResult:
        """The function evaluator callback signature.

        The first argument of the function should be a matrix where each column is a
        variable vector. Depending on the information passed by the second argument, all
        objective and constraint functions for all vectors or for a subset are to be
        calculated.

        The second argument is an [`EvaluatorContext`][ropt.evaluator.EvaluatorContext]
        object, providing supplementary information to the evaluation function.

        Args:
            variables: The matrix of variables to evaluate
            context:   The evaluation context

        Returns:
            An evaluation results object.
        """
