from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(slots=True)
class FunctionEvaluations(ResultField):
    """Per-realization objective and constraint values for an evaluation batch.

    See [Working with Results](../running/results.md) for usage details, and
    [User-defined axes](../running/results.md#user-defined-axes) for metadata
    entries that carry an axis of their own.

    Attributes:
        objectives:  Objective values per realization, shape $(n_r, n_o)$, with
                     axes [`REALIZATION`][ropt.enums.AxisName.REALIZATION] and
                     [`OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE].
        constraints: Nonlinear constraint values per realization, shape
                     $(n_r, n_c)$, with axes
                     [`REALIZATION`][ropt.enums.AxisName.REALIZATION] and
                     [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT].
                     `None` unless nonlinear constraints are configured.
        metadata:    Optional per-realization metadata from the evaluator. Each
                     entry is an array of shape $(n_r,)$ of any `numpy` dtype,
                     including objects; an entry of arrays instead has shape
                     $(n_r, n_k)$ and a second axis named after its key.
    """

    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.OBJECTIVE,
            ),
        },
    )
    constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    metadata: dict[str, NDArray[Any]] = field(
        default_factory=dict,
        metadata={
            "__axes__": (AxisName.REALIZATION,),
        },
    )

    def __post_init__(self) -> None:
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def create(
        cls,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
        metadata: dict[str, NDArray[Any]] | None = None,
    ) -> FunctionEvaluations:
        """Create a `FunctionEvaluations` object with the given data.

        Args:
            objectives:      The objective functions for each realization.
            constraints:     The constraint functions for each realization.
            metadata: Optional info for each evaluation.

        Returns:
            A new FunctionEvaluations object.
        """
        return FunctionEvaluations(
            objectives=objectives,
            constraints=constraints,
            metadata={} if metadata is None else metadata,
        )
