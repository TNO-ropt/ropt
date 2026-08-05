"""Adapt user objective callbacks to the low-level evaluation protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

import numpy as np

from ropt.workflow.evaluators import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

ObjectiveResult = Union[
    EvaluationFunctionResult, float, "Sequence[float]", "NDArray[np.float64]"
]


class ObjectiveCallback(Protocol):
    """The call signature for a high-level objective callback."""

    def __call__(
        self,
        variables: NDArray[np.float64],
        context: EvaluationFunctionContext,
        /,
    ) -> ObjectiveResult:
        """Evaluate objectives and constraints for a single variable vector.

        Args:
            variables: The 1-D variable vector for this evaluation.
            context:   The context identifying the evaluation.

        Returns:
            An [`EvaluationFunctionResult`][ropt.workflow.evaluators.EvaluationFunctionResult],
            a scalar, or a flat sequence of objectives followed by constraints.
        """


def adapt_objective(
    objective: ObjectiveCallback, n_obj: int, n_con: int
) -> EvaluationFunctionCallback:
    """Wrap a user objective so it conforms to the low-level callback protocol.

    Args:
        objective: The user-supplied objective callback.
        n_obj:     The number of objectives expected by the configuration.
        n_con:     The number of nonlinear constraints expected.

    Returns:
        A callback returning an `EvaluationFunctionResult` for every evaluation.
    """

    def _callback(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        return _coerce(objective(variables, context), n_obj, n_con)

    return _callback


def _coerce(
    result: ObjectiveResult, n_obj: int, n_con: int
) -> EvaluationFunctionResult:
    if isinstance(result, EvaluationFunctionResult):
        return result

    total = n_obj + n_con
    array = np.asarray(result, dtype=np.float64)
    if array.ndim == 0:
        if total != 1:
            msg = (
                "A scalar objective result is only allowed with a single objective "
                f"and no constraints, but the configuration expects {n_obj} "
                f"objective(s) and {n_con} constraint(s)."
            )
            raise ValueError(msg)
        array = array.reshape(1)

    if array.shape != (total,):
        msg = (
            f"The objective result must have shape ({total},) "
            f"(objectives first, then constraints), but got shape {array.shape}."
        )
        raise ValueError(msg)

    return EvaluationFunctionResult(
        objectives=array[:n_obj],
        constraints=array[n_obj:] if n_con > 0 else None,
    )
