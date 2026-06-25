"""Defines base classes for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass(slots=True)
class EvaluatorFunctionContext:
    """Context for a single function evaluation.

    Attributes:
        realization:  The realization index.
        perturbation: The perturbation index (`-1` when unperturbed).
        batch_id:     Integer identifying the current evaluation batch.
        eval_idx:     Row index within the batch.
    """

    realization: int
    perturbation: int
    batch_id: int
    eval_idx: int


@dataclass(slots=True)
class EvaluatorFunctionResult:
    """Result of a single function evaluation.

    Attributes:
        objectives:      The objective values as an array.
        constraints:     Optional constraint values as an array.
        evaluation_info: Optional dictionary containing additional information
                         about the evaluation.
    """

    objectives: NDArray[np.float64] | float
    constraints: NDArray[np.float64] | float | None = None
    evaluation_info: dict[str, Any] | None = None


class EvaluatorFunctionCallback(Protocol):
    """Defines the call signature for function callbacks.

    A function following this protocol is called once per active row of the
    evaluation batch, receiving the variable vector for that row together with
    a `EvaluatorFunctionContext` object that identifies the evaluation.

    The function should return a `EvaluatorFunctionResult` object containing the
    evaluation results.
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        context: EvaluatorFunctionContext,
    ) -> EvaluatorFunctionResult:
        """Evaluate objectives and constraints for a single variable vector.

        Args:
            variables:    1-D variable vector for this evaluation.
            context:      The `EvaluatorFunctionContext` object identifying the evaluation.

        Returns:
            The evaluation result as a `EvaluatorFunctionResult` object.
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
