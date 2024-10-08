"""This file defines the event objects that steps may use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ContextUpdate:
    """Base class for data objects used for updating context objects.

    This class is used as a base for classes that store data produced by
    optimization plan steps und used to update context objects.

    Attributes:
        step_name: Optional name of the step generating the data object
    """

    step_name: Optional[str]


@dataclass
class ContextUpdateDict(ContextUpdate):
    """Context update data class used to represent dictionary data.

    Attributes:
        data: Generic data represented by a dictionary.
    """

    data: Dict[str, Any]
