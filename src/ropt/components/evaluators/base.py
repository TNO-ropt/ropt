"""Defines base classes for evaluators."""

from __future__ import annotations

import functools
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ropt.components._transferred import _make_placeholder
from ropt.evaluation import EvaluationBatchResult
from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluation import EvaluationBatchContext


class Evaluator(ABC):
    """Abstract base class for evaluator components within an optimization workflow.

    Subclasses must implement the abstract
    [`eval`][ropt.components.evaluators.Evaluator.eval] method, which is
    responsible for performing the actual evaluation of variables using an
    [`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext] and
    returning an
    [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult].

    Note:
        Evaluators are not safe for concurrent use. An evaluator raises a
        `RuntimeError` if two threads execute its `eval` method at the same
        time. Serial reuse is allowed: the same instance may be reused by
        several compute steps, including on different threads, as long as each
        call fully completes before the next begins. For parallel workflows use
        a dispatching evaluator such as
        [`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator], which
        dispatches tasks to an executor rather than sharing an evaluator across
        threads. See [Optimization Workflows](../workflows/workflows.md#evaluators)
        for usage and pitfalls.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:  # ruff: ignore[undocumented-magic-method]
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
                _orig: Any = original,  # ruff: ignore[any-type]
            ) -> EvaluationBatchResult:
                with self._owner_lock:
                    if self._in_use:
                        msg = "The evaluator is already running on another thread."
                        raise WorkflowError(msg)
                    self._in_use = True
                try:
                    result = _orig(self, variables, context)
                finally:
                    with self._owner_lock:
                        self._in_use = False
                assert isinstance(result, EvaluationBatchResult)
                return result

            cls.eval = _guarded  # type: ignore[method-assign]

    def __init__(self) -> None:
        """Initialize the Evaluator."""
        self._in_use = False
        self._owner_lock = threading.Lock()

    def __reduce__(self) -> tuple[object, tuple[str]]:  # ruff: ignore[undocumented-magic-method]
        return (_make_placeholder, ("An evaluator",))

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
            WorkflowError: If another thread is executing this evaluator's `eval`
                          method at the same time.
        """


@dataclass(slots=True)
class EvaluationFunctionContext:
    """Context for a single function evaluation.

    Attributes:
        realization:  The realization index.
        perturbation: The perturbation index (`-1` when unperturbed).
        batch_id:     Integer identifying the current evaluation batch.
        eval_idx:     Row index within the batch.
        metadata:     The metadata the run was started with, if any.
    """

    realization: int
    perturbation: int
    batch_id: int
    eval_idx: int
    metadata: dict[str, Any] | None = None


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
