from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from ropt.enums import AxisName

from ._result_field import ResultField
from ._results import Results
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._constraint_info import ConstraintInfo
    from ._function_evaluations import FunctionEvaluations
    from ._functions import Functions
    from ._realizations import Realizations


TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class ScaledFunctionResults(ResultField):
    """The scaled counterpart of the fields that have two domains.

    Each field mirrors the field of the same name on
    [`FunctionResults`][ropt.results.FunctionResults], expressed in the domain
    the optimizer works in.

    **Result descriptions**

    === "Variables"

        `variables`: The vector of variable values in the optimizer's domain:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    Attributes:
        variables:       The variable vector the optimizer proposed.
        functions:       Scaled aggregates, or `None` if all realizations failed.
        constraint_info: Constraint differences in the optimizer's domain.
    """

    variables: NDArray[np.float64] = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    functions: Functions | None = None
    constraint_info: ConstraintInfo | None = None

    def __post_init__(self) -> None:
        self.variables = _immutable_copy(self.variables)


@dataclass(slots=True)
class FunctionResults(Results):
    """Results of a function evaluation batch.

    Fields that have two domains appear twice: once here, in the domain that was
    configured, and once under `scaled`, in the domain the optimizer works in.
    The `target_objective` is the exception. It is a weighted total over
    objectives that may differ in both scale and direction, so it exists only in
    the optimizer's domain and has no scaled counterpart.

    See [Working with Results](../running/results.md) for usage details.

    **Result descriptions**

    === "Variables"

        `variables`: The vector of variable values at which the functions were
        evaluated:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Target Objective"

        `target_objective`: The single weighted value the optimizer minimizes:

        - Shape: $()$ — a zero-dimensional array.
        - Axis types: none.

    Attributes:
        variables:        The variable vector that was evaluated.
        evaluations:      Per-realization values returned by the evaluator.
        realizations:     Realization activity and weights.
        functions:        Aggregated function values, or `None` if all failed.
        target_objective: The value the optimizer minimizes, in its own domain.
        scaled:           The same quantities as the optimizer works with them.
        constraint_info:  Constraint differences and violations, if applicable.
    """

    variables: NDArray[np.float64] = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    evaluations: FunctionEvaluations
    realizations: Realizations
    functions: Functions | None
    target_objective: NDArray[np.float64] | None
    scaled: ScaledFunctionResults
    constraint_info: ConstraintInfo | None = None

    def __post_init__(self) -> None:
        self.variables = _immutable_copy(self.variables)
        self.target_objective = _immutable_copy(self.target_objective)
        assert (self.target_objective is None) == (self.functions is None)
