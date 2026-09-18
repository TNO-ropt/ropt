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

    See [Working with Results](../running/results.md) for usage details.

    **Result descriptions**

    === "Objectives"

        `objectives`: The calculated objective function values for each
        realization. This is a two-dimensional array of floating point values
        where each row corresponds to a realization and each column corresponds
        to an objective:

        - Shape $(n_r, n_o)$, where:
            - $n_r$ is the number of realizations.
            - $n_o$ is the number of objectives.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

    === "Constraints"

        `constraints`: The calculated constraint function values for each
        realization. Only provided if non-linear constraints are defined. This
        is a two-dimensional array of floating point values where each row
        corresponds to a realization and each column corresponds to a
        constraint:

        - Shape $(n_r, n_c)$, where:
            - $n_r$ is the number of realizations.
            - $n_c$ is the number of constraints.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

    === "Metadata"

        `metadata`: Optional metadata associated with each realization,
        potentially provided by the evaluator. Each value is a one-dimensional
        array of any type supported by `numpy`, including objects, so each key
        may carry its own dtype:

        - Shape: $(n_r,)$, where:
            - $n_r$ is the number of realizations.
        - Axis type:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]

        An entry whose values are arrays rather than scalars has shape
        $(n_r, n_k)$ and carries a second, user-defined axis named after its
        key. See [User-defined axes](../running/results.md#user-defined-axes).

    Attributes:
        objectives:  The objective function values for each realization.
        constraints: The constraint function values for each realization.
        metadata:    Optional metadata for each evaluated realization.
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
