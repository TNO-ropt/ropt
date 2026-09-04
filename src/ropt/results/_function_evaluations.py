from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ropt._scaling import value_from_optimizer
from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.context import EnOptContext


@dataclass(slots=True)
class FunctionEvaluations(ResultField):
    """Per-realization objective and constraint values for an evaluation batch.

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    **Result descriptions**

    === "Variables"

        `variables`: The vector of variable values at which the functions
        were evaluated:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

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
        potentially provided by the evaluator. If provided, each value in the
        metadata dictionary must be a one-dimensional array of arbitrary type
        supported by `numpy` (including objects):

        - Shape: $(n_r,)$, where:
            - $n_r$ is the number of realizations.
        - Axis type:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]

    Note: Metadata data type.
        The data type of the metadata fields is not fixed. Each field in the
        `metadata` dictionary can have its own data type, which must be a
        one-dimensional array of any type supported by `numpy`, including object
        arrays. This allows for maximum flexibility in the kind of metadata that
        can be included, such as strings, integers, floats, or even complex
        objects.

    Attributes:
        variables:   The variable vector.
        objectives:  The objective function values for each realization.
        constraints: The constraint function values for each realization.
        metadata:    Optional metadata for each evaluated realization.
    """

    variables: NDArray[np.float64] = field(
        metadata={
            "__axes__": (AxisName.VARIABLE,),
        },
    )
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
        self.variables = _immutable_copy(self.variables)
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def create(
        cls,
        variables: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
        metadata: dict[str, NDArray[Any]] | None = None,
    ) -> FunctionEvaluations:
        """Create a `FunctionEvaluations` object with the given data.

        Args:
            variables:       The unperturbed variable vector.
            objectives:      The objective functions for each realization.
            constraints:     The constraint functions for each realization.
            metadata: Optional info for each evaluation.

        Returns:
            A new FunctionEvaluations object.
        """
        return FunctionEvaluations(
            variables=variables,
            objectives=objectives,
            constraints=constraints,
            metadata={} if metadata is None else metadata,
        )

    def _transform_from_optimizer(self, context: EnOptContext) -> FunctionEvaluations:
        variables = value_from_optimizer(
            self.variables, context.variables.scales, context.variables.offsets
        )
        objectives = value_from_optimizer(
            self.objectives, context.get_objective_scales()
        )
        constraints = self.constraints
        if constraints is not None:
            constraint_scales = context.get_constraint_scales()
            assert constraint_scales is not None
            constraints = value_from_optimizer(constraints, constraint_scales)

        return FunctionEvaluations(
            variables=variables,
            objectives=objectives,
            constraints=constraints,
            metadata=self.metadata,
        )
