from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_PERTURBED_AXES = (
    AxisName.REALIZATION,
    AxisName.PERTURBATION,
    AxisName.VARIABLE,
)


@dataclass(slots=True)
class GradientPerturbations(ResultField):
    """Perturbed variables and their evaluated values for a gradient batch.

    See [Working with Results](../results/results.md) for usage details.

    **Result descriptions**

    === "Active"

        `active`: A boolean array indicating which realizations were evaluated.
        `True` indicates that a realization was evaluated:

        - Shape $(n_r,)$, where:
            - $n_r$ is the number of realizations.
        - Axis type:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]

    === "Variables"

        `variables`: A three-dimensional array of perturbed variable values for
        each realization and perturbation:

        - Shape $(n_r, n_p, n_v)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_v$ is the number of variables.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Objectives"

        `objectives`: A three-dimensional array of objective values for each
        realization and perturbation:

        - Shape $(n_r, n_p, n_o)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_o$ is the number of objectives.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

    === "Constraints"

        `constraints`: A three-dimensional array of non-linear constraint
        values for each realization and perturbation:

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
        [User-defined axes](../results/results.md#user-defined-axes).

    Attributes:
        active:      Boolean array indicating evaluated realizations.
        variables:   The perturbed vectors that were evaluated.
        objectives:  Objective values per realization and perturbation.
        constraints: Constraint values per realization and perturbation.
        metadata:    Optional metadata per realization and perturbation.
    """

    active: NDArray[np.bool_] = field(
        metadata={
            "__axes__": (AxisName.REALIZATION,),
        },
    )
    variables: NDArray[np.float64] = field(
        metadata={"__axes__": _PERTURBED_AXES},
    )
    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
                AxisName.OBJECTIVE,
            ),
        },
    )
    constraints: NDArray[np.float64] | None = field(
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
        self.active = _immutable_copy(self.active)
        self.variables = _immutable_copy(self.variables)
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def create(
        cls,
        active: NDArray[np.bool_],
        variables: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
        metadata: dict[str, NDArray[Any]] | None = None,
    ) -> GradientPerturbations:
        """Create a `GradientPerturbations` object with the given data.

        Args:
            active:      Which realizations were evaluated.
            variables:   The perturbed vectors that were evaluated.
            objectives:  Objective values per realization and perturbation.
            constraints: Constraint values per realization and perturbation.
            metadata:    Optional info for each evaluation.

        Returns:
            A new `GradientPerturbations` object.
        """
        return GradientPerturbations(
            active=active,
            variables=variables,
            objectives=objectives,
            constraints=constraints,
            metadata={} if metadata is None else metadata,
        )


@dataclass(slots=True)
class ScaledGradientPerturbations(ResultField):
    """The scaled counterpart of the perturbed variables.

    Mirrors the `variables` field of
    [`GradientPerturbations`][ropt.results.GradientPerturbations], expressed
    in the domain the optimizer works in.

    **Result descriptions**

    === "Variables"

        `variables`: A three-dimensional array of perturbed variable values for
        each realization and perturbation:

        - Shape: $(n_r, n_p, n_v)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_v$ is the number of variables.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    Attributes:
        variables: The perturbed vectors in the optimizer's domain.
    """

    variables: NDArray[np.float64] = field(
        metadata={"__axes__": _PERTURBED_AXES},
    )

    def __post_init__(self) -> None:
        self.variables = _immutable_copy(self.variables)
