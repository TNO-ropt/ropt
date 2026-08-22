"""Adapt user evaluation functions to the low-level evaluation protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

import numpy as np

from ropt.components.evaluators import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

FunctionValue = Union[
    EvaluationFunctionResult, float, "Sequence[float]", "NDArray[np.float64]"
]


class EvaluationFunction(Protocol):
    """The call signature for a high-level evaluation function."""

    def __call__(
        self,
        variables: NDArray[np.float64],
        context: EvaluationFunctionContext,
        /,
    ) -> FunctionValue:
        """Evaluate the objectives and constraints for a single variable vector.

        Args:
            variables: The 1-D variable vector for this evaluation.
            context:   The context identifying the evaluation.

        Returns:
            An [`EvaluationFunctionResult`][ropt.components.evaluators.EvaluationFunctionResult],
            a scalar, or a flat sequence of objectives followed by constraints.
        """


def adapt_function(
    function: EvaluationFunction, n_obj: int, n_con: int
) -> EvaluationFunctionCallback:
    """Wrap a user evaluation function to conform to the low-level protocol.

    Args:
        function: The user-supplied evaluation function.
        n_obj:    The number of objectives expected by the configuration.
        n_con:    The number of nonlinear constraints expected.

    Returns:
        A callback returning an `EvaluationFunctionResult` for every evaluation.
    """
    return _AdaptedFunction(function, n_obj, n_con)


class _AdaptedFunction:
    # A class rather than a closure, so that a process or HPC executor can
    # serialize it with the standard pickle module.

    def __init__(self, function: EvaluationFunction, n_obj: int, n_con: int) -> None:
        self._function = function
        self._n_obj = n_obj
        self._n_con = n_con

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        return _coerce(self._function(variables, context), self._n_obj, self._n_con)


def _coerce(result: FunctionValue, n_obj: int, n_con: int) -> EvaluationFunctionResult:
    # The shapes the configuration expects are known here, so a function may
    # return a bare number or a flat sequence and still be checked properly.
    if isinstance(result, EvaluationFunctionResult):
        return result

    total = n_obj + n_con
    array = np.asarray(result, dtype=np.float64)
    if array.ndim == 0:
        if total != 1:
            msg = (
                "A scalar return value is only allowed with a single objective "
                f"and no constraints, but the configuration expects {n_obj} "
                f"objective(s) and {n_con} constraint(s)."
            )
            raise ValueError(msg)
        array = array.reshape(1)

    if array.shape != (total,):
        msg = (
            f"The evaluation function must return a value of shape ({total},) "
            f"(objectives first, then constraints), but got shape {array.shape}."
        )
        raise ValueError(msg)

    return EvaluationFunctionResult(
        objectives=array[:n_obj],
        constraints=array[n_obj:] if n_con > 0 else None,
    )
