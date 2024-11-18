from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig


@dataclass(slots=True)
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
        active:             Signifies which realizations are active.
    """

    config: EnOptConfig
    realizations: NDArray[np.intc]
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
    constraints: NDArray[np.float64] | None = None
    batch_id: int | None = None
    evaluation_ids: NDArray[np.intc] | None = None


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
        r"""The function evaluator callback signature.

        The first argument of the function should be a matrix where each column is a
        variable vector. Depending on the information passed by the second argument, all
        objective and constraint functions for all vectors or for a subset are to be
        calculated.

        The second argument is an [`EvaluatorContext`][ropt.evaluator.EvaluatorContext]
        object that provides supplementary information to the evaluation function.

        The return value should be an [`EvaluatorResult`][ropt.evaluator.EvaluatorResult]
        object containing the calculated values of the objective and constraint functions for
        all variable vectors and realizations, along with any additional metadata.

        Args:
            variables: The matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            An evaluation results object.

        Tip: Reusing Objective and Non-linear Constraint Values
            When defining multiple objectives, there may be a need to reuse the
            same objective value multiple times. For instance, a total objective could
            consist of the mean of the objectives for each realization, plus the
            standard deviation of the same values. This can be implemented by defining
            two objectives: the first calculated as the mean of the realizations, and
            the second using a function transform to compute the standard deviations.
            The optimizer is unaware that both objectives use the same set of
            realizations. To prevent redundant calculations, the evaluator should
            compute the results of the realizations once and return them for both
            objectives.

            Non-linear constraint values may potentially appear multiple times in the
            `constraint_results` matrix. For example, to express the constraint \(a < F_c \le b\),
            two constraints must be defined: \(F_c \ge a\) and \(F_c \le b\),
            sharing the same function value \(F_c\) but differing in types and
            right-hand sides (`ConstraintType.GE`/`ConstraintType.LE` and \(a\)/\(b\)). The
            run method should ensure that the calculated value of \(F_c\) is the same in
            both cases, which is most efficiently achieved by evaluating \(F_c\) only
            once.
        """
