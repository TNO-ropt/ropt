"""Defines base classes for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult


class Evaluator(ABC):
    """Abstract base class for evaluator components within an optimization workflow.

    Subclasses must implement the abstract
    [`eval`][ropt.workflow.evaluators.Evaluator.eval] method, which is
    responsible for performing the actual evaluation of variables using an
    [`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext] and
    returning an
    [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult].
    """

    @abstractmethod
    def eval(
        self, variables: NDArray[np.float64], context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate objective and constraint functions for given variables.

        This follows the [`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback] protocol.

        Args:
            variables: The matrix of variables to evaluate. Each row represents
                       a variable vector.
            context:   The evaluation context, providing additional information
                       about the evaluation.

        Returns:
            An evaluation results object containing the calculated values.
        """


class FunctionCallback(Protocol):
    """Defines the call signature for function callbacks.

    A function following this protocol is called once per active row of the
    evaluation batch, receiving the variable vector for that row together with
    keyword arguments that identify the evaluation.

    The function should return either:

    - A 1-D NumPy array of length `n_objectives + n_constraints`.
    - A dictionary with a `"result"` key containing that array; any
      additional keys are stored as `evaluation_info` entries.
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        /,
        *,
        realization: int,
        perturbation: int,
        batch_id: int,
        eval_idx: int,
    ) -> NDArray[np.float64] | dict[str, Any]:
        """Evaluate objectives and constraints for a single variable vector.

        Args:
            variables:    1-D variable vector for this evaluation.
            realization:  The realization index.
            perturbation: The perturbation index (`-1` when unperturbed).
            batch_id:     Integer identifying the current evaluation batch.
            eval_idx:     Row index within the batch.

        Returns:
            The evaluation result as an array or a dictionary.
        """


class NameCallback(Protocol):
    """Defines the call signature for callbacks to get the name of an evaluation."""

    def __call__(
        self,
        realization: int,
        perturbation: int,
        batch_id: int,
        eval_idx: int,
    ) -> str:
        """Get the name for a single evaluation.

        Args:
            realization:  The realization index.
            perturbation: The perturbation index (`-1` when unperturbed).
            batch_id:     Integer identifying the current evaluation batch.
            eval_idx:     Row index within the batch.

        Returns:
            The name of the evaluation.
        """
