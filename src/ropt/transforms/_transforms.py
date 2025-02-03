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
    objectives: ObjectiveTransform | None = None
    nonlinear_constraints: NonLinearConstraintTransform | None = None
