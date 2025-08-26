"""This module defines the OptModelTransforms class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from .base import (
        NonLinearConstraintTransform,
        ObjectiveTransform,
        VariableTransform,
    )


@dataclass
class OptModelTransforms:
    """A container for optimization model transformers."""

    variables: VariableTransform | None = None
    """A `VariableTransform` object that defines the transformation for
    variables.

    If `None`, no transformation is applied to variables.
    """
    objectives: ObjectiveTransform | None = None
    """An `ObjectiveTransform` object that defines the transformation for
    objectives.

    If `None`, no transformation is applied to objectives.
    """
    nonlinear_constraints: NonLinearConstraintTransform | None = None
    """A `NonLinearConstraintTransform` object that defines the transformation
    for nonlinear constraints.

    If `None`, no transformation is applied to nonlinear constraints.
    """

    objective_weights: NDArray[np.float64] | None = None
    """Objective weights that are used by the objective transform.

    May be `None` if `objectives` is `None`.

    The weights will be normalized to sum to 1.
    """

    def __post_init__(self) -> None:
        if self.objective_weights is not None:
            self.objective_weights = (
                self.objective_weights / self.objective_weights.sum()
            )
            self.objective_weights.setflags(write=False)
