"""Defines base classes for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult


class Evaluator(ABC):
    """Abstract base class for evaluator components within an optimization workflow.

    Subclasses must implement the abstract
    [`eval`][ropt.workflow.evaluators.Evaluator.eval] method, which is responsible
    for performing the actual evaluation of variables using an
    [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] and returning an
    [`EvaluatorResult`][ropt.evaluator.EvaluatorResult].
    """

    @abstractmethod
    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate objective and constraint functions for given variables.

        Args:
            variables: The matrix of variables to evaluate. Each row represents
                       a variable vector.
            context:   The evaluation context, providing additional information
                       about the evaluation.

        Returns:
            An evaluation results object containing the calculated values.
        """
