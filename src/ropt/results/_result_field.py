from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ropt.enums import ResultAxis

TypeResultField = TypeVar("TypeResultField", bound="ResultField")


@dataclass(slots=True)
class ResultField:
    """Base class for fields within `Results` objects.

    The `ResultField` class serves as a foundation for defining the various data
    fields that can be stored within [`Results`][ropt.results.Results] objects.
    These fields typically hold multi-dimensional numerical data, such as
    objective values, constraint values, or gradients.

    This class provides a standardized way to:

    * Store metadata about the axes of multi-dimensional arrays.
    * Retrieve the axes associated with a specific field.

    Derived classes, such as
    [`FunctionEvaluations`][ropt.results.FunctionEvaluations] or
    [`Gradients`][ropt.results.Gradients], extend this base class to define
    specific data structures for different types of optimization results.
    """

    @classmethod
    def get_axes(cls, name: str) -> tuple[ResultAxis, ...]:
        """Retrieve the axes associated with a specific field.

        Fields within a `ResultField` object that store multi-dimensional
        `numpy` arrays, contain metadata that describes the meaning of each
        dimension in the array. This method retrieves the axes of a field within
        a ResultField object from that meta-data, returning a tuple of
        `ResultAxis`][ropt.enums.ResultAxis] enums.

        Args:
            name: The name of the field (sub-field) within the
                  `ResultField` instance or class.

        Raises:
            ValueError: If the provided field name is not recognized.

        Returns:
            A tuple of [`ResultAxis`][ropt.enums.ResultAxis] enums, representing the axes of the field.
        """
        metadata = next(
            (item.metadata for item in fields(cls) if item.name == name), None
        )
        if metadata is None:
            msg = f"Unknown field name: {name}"
            raise ValueError(msg)
        return metadata.get("__axes__", ())
