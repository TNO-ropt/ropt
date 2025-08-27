"""This module defines the OptModelTransforms class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
