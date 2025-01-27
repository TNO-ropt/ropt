"""This module defines the Transforms class, a container for transforms."""

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
class Transforms:
    """A container for transformers."""

    variables: VariableTransform | None = None
    objectives: ObjectiveTransform | None = None
    nonlinear_constraints: NonLinearConstraintTransform | None = None
