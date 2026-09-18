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
class GradientEvaluations(ResultField):
    """Per-realization evaluation data for perturbed variables.

    See [Working with Results](../running/results.md) for usage details, and
    [User-defined axes](../running/results.md#user-defined-axes) for metadata
    entries that carry an axis of their own.

    Attributes:
        perturbed_objectives:  Objective values per realization and
                               perturbation, shape $(n_r, n_p, n_o)$, with axes
                               [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
                               [`PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
                               and [`OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE].
        perturbed_constraints: Nonlinear constraint values per realization and
                               perturbation, shape $(n_r, n_p, n_c)$, with axes
                               [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
                               [`PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
                               and
                               [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT].
                               `None` unless nonlinear constraints are
                               configured.
        metadata:              Optional metadata from the evaluator. Each entry
                               is an array of shape $(n_r, n_p)$ of any `numpy`
                               dtype, including objects; an entry of arrays
                               instead has shape $(n_r, n_p, n_k)$ and a third
                               axis named after its key.
    """

    perturbed_objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
                AxisName.OBJECTIVE,
            ),
        },
    )
    perturbed_constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
                AxisName.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    metadata: dict[str, NDArray[Any]] = field(
        default_factory=dict,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
            ),
        },
    )

    def __post_init__(self) -> None:
        self.perturbed_objectives = _immutable_copy(self.perturbed_objectives)
        self.perturbed_constraints = _immutable_copy(self.perturbed_constraints)

    @classmethod
    def create(
        cls,
        perturbed_objectives: NDArray[np.float64],
        perturbed_constraints: NDArray[np.float64] | None = None,
        metadata: dict[str, NDArray[Any]] | None = None,
    ) -> GradientEvaluations:
        """Create a `GradientEvaluations` object with the given data.

        Args:
            perturbed_objectives:  Objective function values for each
                                   realization and perturbation.
            perturbed_constraints: Constraint function values for each
                                   realization and perturbation.
            metadata:              Optional info for each evaluation.

        Returns:
            A new `GradientEvaluations` object.
        """
        return GradientEvaluations(
            perturbed_objectives=perturbed_objectives,
            perturbed_constraints=perturbed_constraints,
            metadata={} if metadata is None else metadata,
        )
