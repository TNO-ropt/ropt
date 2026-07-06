"""Defines base classes for evaluators."""

from __future__ import annotations

import functools
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ropt.evaluation import EvaluationBatchResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluation import EvaluationBatchContext


class Evaluator(ABC):
    """Abstract base class for evaluator components within an optimization workflow.

    Subclasses must implement the abstract
    [`eval`][ropt.workflow.evaluators.Evaluator.eval] method, which is
    responsible for performing the actual evaluation of variables using an
    [`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext] and
    returning an
    [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult].

    Warning:
        Evaluator instances must not be called concurrently from multiple
        threads. For parallel workflows use a server-based evaluator such as
        [`AsyncEvaluator`][ropt.workflow.evaluators.AsyncEvaluator].
    """

    def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)
        if "eval" in cls.__dict__ and not getattr(
            cls.__dict__["eval"], "__wrapped__", None
        ):
            original = cls.__dict__["eval"]

            @functools.wraps(original)
            def _guarded(
                self: Evaluator,
                variables: NDArray[np.float64],
                context: EvaluationBatchContext,
                *,
                _orig: Any = original,  # noqa: ANN401
            ) -> EvaluationBatchResult:
                if not self._in_use.acquire(blocking=False):
                    msg = (
                        "Evaluator does not support concurrent use across threads; "
                        "use a server-based evaluator (e.g. AsyncEvaluator) for parallel workflows."
                    )
                    raise RuntimeError(msg)
                try:
                    result = _orig(self, variables, context)
                    assert isinstance(result, EvaluationBatchResult)
                    return result
                finally:
                    self._in_use.release()

            cls.eval = _guarded  # type: ignore[method-assign]

    def __init__(self) -> None:
        """Initialize the Evaluator."""
        self._in_use = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:  # noqa: D105
        state = self.__dict__.copy()
        state.pop("_in_use", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:  # noqa: D105
        self.__dict__.update(state)
        self._in_use = threading.Lock()

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

        Raises:
            RuntimeError: If this evaluator is already in use by another thread.
        """


@dataclass(slots=True)
class EvaluationFunctionContext:
    """Context for a single function evaluation.

    Attributes:
        realization:  The realization index.
        perturbation: The perturbation index (`-1` when unperturbed).
        batch_id:     Integer identifying the current evaluation batch.
        eval_idx:     Row index within the batch.
        name:         Optional task name set by the evaluator.
    """

    realization: int
    perturbation: int
    batch_id: int
    eval_idx: int
    name: str | None = None


@dataclass(slots=True)
class EvaluationFunctionResult:
    """Result of a single function evaluation.

    Attributes:
        objectives:  The objective values as an array.
        constraints: Optional constraint values as an array.
        metadata:    Optional dictionary containing additional information
                     about the evaluation.
    """

    objectives: NDArray[np.float64] | float
    constraints: NDArray[np.float64] | float | None = None
    metadata: dict[str, Any] | None = None


class EvaluationFunctionCallback(Protocol):
    """Defines the call signature for function callbacks.

    A function following this protocol is called once per active row of the
    evaluation batch, receiving the variable vector for that row together with
    a `EvaluationFunctionContext` object that identifies the evaluation.

    The function should return a `EvaluationFunctionResult` object containing the
    evaluation results.
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        context: EvaluationFunctionContext,
    ) -> EvaluationFunctionResult:
        """Evaluate objectives and constraints for a single variable vector.

        Args:
            variables:    1-D variable vector for this evaluation.
            context:      The `EvaluationFunctionContext` object identifying the evaluation.

        Returns:
            The evaluation result as a `EvaluationFunctionResult` object.
        """


class NameCallback(Protocol):
    """Defines the call signature for callbacks to get the name of a task."""

    def __call__(self, contexts: Sequence[EvaluationFunctionContext]) -> str:
        """Get the name for a task.

        The task may contain a single evaluation or a bundle of several
        evaluations that the worker runs sequentially. The callback receives
        the `EvaluationFunctionContext` objects for every evaluation in the
        task, in submission order, and should return a single string used as
        the task name.

        Args:
            contexts: The contexts for every evaluation in the task.

        Returns:
            The name of the task.
        """
