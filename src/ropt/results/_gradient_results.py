from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt.enums import AxisName

from ._result_field import ResultField
from ._results import Results
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._function_results import FunctionResults
    from ._gradient_aggregates import GradientAggregates
    from ._gradient_estimates import GradientEstimates
    from ._gradient_perturbations import (
        GradientPerturbations,
        ScaledGradientPerturbations,
    )
    from ._realizations import Realizations


@dataclass(slots=True)
class ScaledGradientResults(ResultField):
    """The scaled counterpart of the fields that have two domains.

    Each field mirrors the field of the same name on
    [`GradientResults`][ropt.results.GradientResults], expressed in the domain
    the optimizer works in.

    **Result descriptions**

    === "Variables"

        `variables`: The vector of unperturbed variable values in the
        optimizer's domain:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    Attributes:
        variables:     The variable vector the optimizer proposed.
        perturbations: The perturbed vectors in the optimizer's domain.
        estimates:     Scaled per-realization gradients, or `None`.
        aggregates:    Scaled gradients, or `None` if estimation failed.
    """

    variables: NDArray[np.float64] = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    perturbations: ScaledGradientPerturbations
    estimates: GradientEstimates | None = None
    aggregates: GradientAggregates | None = None

    def __post_init__(self) -> None:
        self.variables = _immutable_copy(self.variables)


@dataclass(slots=True)
class GradientResults(Results):
    """Results of a gradient evaluation batch.

    Fields that have two domains appear twice: once here, in the domain that was
    configured, and once under `scaled`, in the domain the optimizer works in.
    The `target_objective` is the exception. It is the gradient of a weighted
    total over objectives that may differ in both scale and direction, with
    respect to the scaled variables, so it exists only in the optimizer's domain
    and has no scaled counterpart.

    See [Working with Results](../results/results.md) for usage details.

    **Result descriptions**

    === "Variables"

        `variables`: The vector of unperturbed variable values:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Target Objective"

        `target_objective`: The gradient of the target objective, which the
        optimizer descends:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    Attributes:
        source_key:       The batch and function ID of a function evaluation
                          whose values were used to construct this gradient.
                          Usually, but not necessarily, that function was
                          evaluated at the same point as the gradient.
        variables:        The variable vector that was perturbed.
        perturbations:    The perturbed vectors and the values returned for them.
        estimates:        Per-realization gradients, or `None` if realizations
                          are merged or no gradients were estimated.
        realizations:     Realization weights.
        aggregates:       Aggregated gradients, or `None` if estimation failed.
        target_objective: The gradient the optimizer descends, in its own domain.
        scaled:           The same quantities as the optimizer works with them.
    """

    source_key: tuple[int, int]
    variables: NDArray[np.float64] = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    perturbations: GradientPerturbations
    estimates: GradientEstimates | None
    realizations: Realizations
    aggregates: GradientAggregates | None
    target_objective: NDArray[np.float64] | None = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    scaled: ScaledGradientResults

    def __post_init__(self) -> None:
        self.variables = _immutable_copy(self.variables)
        self.target_objective = _immutable_copy(self.target_objective)
        assert (self.target_objective is None) == (self.aggregates is None)

    def uses(self, function: FunctionResults) -> bool:
        """Return whether this gradient was constructed from a function evaluation.

        Args:
            function: The function evaluation to check.

        Returns:
            Whether the batch and function ID of `function` match `source_key`.
        """
        return self.source_key == (function.batch_id, function.function_id)
