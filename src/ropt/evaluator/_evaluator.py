from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig


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
    - The realization index for each variable vector.
    - The perturbation index for each variable vector (if applicable). A value
      less than 0 indicates that the vector is not a perturbation.
    - Boolean matrices (`active_objectives` and `active_constraints`) indicating
      which objective/realization and constraint/realization evaluations are
      required by the optimizer.
    - A boolean vector (`active`) indicating which realizations require
      evaluation.

    The `active_objectives` and `active_constraints` matrices are structured
    such that each column corresponds to a realization, and each row corresponds
    to a function or constraint. A `True` value signifies that the corresponding
    evaluation is essential for the optimizer.

    Note: The `active` property
        In many cases, evaluators may only be able to compute all objectives and
        constraints for a given realization or none at all. In these scenarios,
        the `active` property provides a simplified view, indicating only the
        realizations that need to be evaluated. `active` cannot be set when creating
        the evaluator context, it is calculated from `active_objectives` and
        `active_constraints`.

    Args:
        config:             Configuration of the optimizer.
        realizations:       Realization numbers for each requested evaluation.
        perturbations:      Perturbation numbers for each requested evaluation.
                            A value less than 0 indicates that the vector is
                            not a perturbation.
        active_objectives:  Indicates which function/realization evaluations are
                            essential for the optimizer.
        active_constraints: Indicates which constraint/realization evaluations
                            are essential for the optimizer.
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
