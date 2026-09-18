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

    See [Working with Results](../running/results.md) for usage details.

    **Result descriptions**

    === "Perturbed Objectives"

        `perturbed_objectives`: A three-dimensional array of perturbed
        calculated objective function values for each realization and
        perturbation:

        - Shape $(n_r, n_p, n_o)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_o$ is the number of objectives.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

    === "Perturbed Constraints"

        `perturbed_constraints`: A three-dimensional array of perturbed
        calculated non-linear constraint values for each realization and
        perturbation:

        - Shape $(n_r, n_p, n_c)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_c$ is the number of constraints.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

    === "Metadata"

        `metadata`: Optional metadata associated with each realization,
        potentially provided by the evaluator. Each value is a two-dimensional
        array of any type supported by `numpy`, including objects, so each key
        may carry its own dtype:

        - Shape: $(n_r, n_p)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]

        An entry whose values are arrays rather than scalars has shape
        $(n_r, n_p, n_k)$ and carries a third, user-defined axis named after
        its key. See
        [User-defined axes](../running/results.md#user-defined-axes).

    Attributes:
        perturbed_objectives:  The objective function values for each
                               realization and perturbation.
        perturbed_constraints: The constraint function values for each
                               realization and perturbation.
        metadata:              Optional metadata for each evaluated
                               realization and perturbation.
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
