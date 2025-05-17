"""This module implements the default function evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ropt.plugins.plan.base import Evaluator, PlanComponent

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult
    from ropt.plan import Plan


class DefaultFunctionEvaluator(Evaluator):
    """An evaluator that forwards calls to an evaluator function.

    This class acts as an adapter, allowing a standard Python callable (which
    matches the signature of the `eval` method) to be used as an
    [`Evaluator`][ropt.plugins.plan.base.Evaluator] within an optimization
    [`Plan`][ropt.plan.Plan].

    It is initialized with an `evaluator` callable. When the `eval` method of
    this class is invoked by the plan, it simply delegates the call, along with
    all arguments, to the wrapped `evaluator` function.

    This is useful for integrating existing evaluation logic that is not already
    structured as an `Evaluator` subclass into a `ropt` plan.

    The `clients` parameter acts as a filter, determining which plan steps this
    evaluator should serve.
    """

    def __init__(
        self,
        plan: Plan,
        tags: set[str] | None = None,
        clients: set[PlanComponent | str] | None = None,
        *,
        evaluator: Callable[[NDArray[np.float64], EvaluatorContext], EvaluatorResult],
    ) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            plan:      The parent plan instance.
            tags:      Optional tags
            clients:   The steps that are allowed to use this evaluator.
            evaluator: The callable that will perform the actual evaluation.
        """
        super().__init__(plan, tags, clients)
        self._evaluator = evaluator

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Forward the evaluation call to the wrapped evaluator function.

        Args:
            variables: The matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.
        """
        return self._evaluator(variables, context)
