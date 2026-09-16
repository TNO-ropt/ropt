from __future__ import annotations

from dataclasses import Field, dataclass, fields
from typing import Any, ClassVar, TypeVar

TypeResultField = TypeVar("TypeResultField", bound="ResultField")


class AxisMetadata:
    """Mixin for dataclasses whose fields carry axis metadata."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @classmethod
    def get_axes(cls, name: str) -> tuple[str, ...]:
        """Return the axis metadata for a named field.

        Args:
            name: The name of the field within this dataclass.

        Returns:
            A tuple of axis names, each an [`AxisName`][ropt.enums.AxisName] value.

        Raises:
            ValueError: If the field name is not recognized.
        """
        metadata = next(
            (item.metadata for item in fields(cls) if item.name == name), None
        )
        if metadata is None:
            msg = f"Unknown field name: {name}"
            raise ValueError(msg)
        axes: tuple[str, ...] = metadata.get("__axes__", ())
        return axes


@dataclass(slots=True)
class ResultField(AxisMetadata):
    """Base class for result field containers that carry axis metadata.

    See [Working with Results](../running/results.md#axes-and-dimensionality) for
    how axis metadata is used.
    """
