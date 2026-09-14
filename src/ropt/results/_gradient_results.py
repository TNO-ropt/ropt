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

    from ._gradient_evaluations import GradientEvaluations
    from ._gradients import Gradients
    from ._realizations import Realizations

TypeResults = TypeVar("TypeResults", bound="Results")

_PERTURBED_AXES = (
    AxisName.REALIZATION,
    AxisName.PERTURBATION,
    AxisName.VARIABLE,
)


@dataclass(slots=True)
class ScaledGradientResults(ResultField):
    """The scaled counterpart of the fields that have two domains.

    Each field mirrors the field of the same name on
    [`GradientResults`][ropt.results.GradientResults], expressed in the domain
    the optimizer works in.

    Attributes:
        variables:           The variable vector the optimizer proposed.
        perturbed_variables: The perturbed vectors in the optimizer's domain.
        gradients:           Scaled gradients, or `None` if estimation failed.
    """

    variables: NDArray[np.float64] = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    perturbed_variables: NDArray[np.float64] = field(
        metadata={"__axes__": _PERTURBED_AXES},
    )
    gradients: Gradients | None = None

    def __post_init__(self) -> None:
        self.variables = _immutable_copy(self.variables)
        self.perturbed_variables = _immutable_copy(self.perturbed_variables)


@dataclass(slots=True)
class GradientResults(Results):
    """Results of a gradient evaluation batch.

    Fields that have two domains appear twice: once here, in the domain that was
    configured, and once under `scaled`, in the domain the optimizer works in.
    The `target_gradient` is the exception. It differentiates a weighted total
    over objectives that may differ in both scale and direction, with respect to
    the scaled variables, so it exists only in the optimizer's domain and has no
    scaled counterpart.

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    Attributes:
        variables:           The variable vector that was perturbed.
        perturbed_variables: The perturbed vectors that were evaluated.
        evaluations:         Per-perturbation values returned by the evaluator.
        realizations:        Realization activity and weights.
        gradients:           Aggregated gradients, or `None` if estimation failed.
        target_gradient:     The gradient the optimizer descends, in its own domain.
        scaled:              The same quantities as the optimizer works with them.
    """

    variables: NDArray[np.float64] = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    perturbed_variables: NDArray[np.float64] = field(
        metadata={"__axes__": _PERTURBED_AXES},
    )
    evaluations: GradientEvaluations
    realizations: Realizations
    gradients: Gradients | None
    target_gradient: NDArray[np.float64] | None = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    scaled: ScaledGradientResults

    def __post_init__(self) -> None:
        self.variables = _immutable_copy(self.variables)
        self.perturbed_variables = _immutable_copy(self.perturbed_variables)
        self.target_gradient = _immutable_copy(self.target_gradient)
        assert (self.target_gradient is None) == (self.gradients is None)
