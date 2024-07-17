"""This file defines the event objects that steps may use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from ropt.results import Results


@dataclass
class ContextUpdate:
    """Base class for data objects used for updating context objects.

    This class is used as a base for classes that store data produced by
    workflow steps und used to update context objects.

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


@dataclass
class ContextUpdateResults(ContextUpdate):
    """Context update data class used for updating with new results.

    Attributes:
        results: The new results
    """

    results: Tuple[Results, ...] = field(default_factory=tuple)
