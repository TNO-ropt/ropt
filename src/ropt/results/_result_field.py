from __future__ import annotations

from dataclasses import Field, dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

if TYPE_CHECKING:
    from ropt.enums import AxisName

TypeResultField = TypeVar("TypeResultField", bound="ResultField")


class AxisMetadata:
    """Mixin for dataclasses whose fields carry axis metadata."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @classmethod
    def get_axes(cls, name: str) -> tuple[AxisName, ...]:
        """Return the axis metadata for a named field.

        Args:
            name: The name of the field within this dataclass.

        Returns:
            A tuple of [`AxisName`][ropt.enums.AxisName] values.

        Raises:
            ValueError: If the field name is not recognized.
        """
        metadata = next(
            (item.metadata for item in fields(cls) if item.name == name), None
        )
        if metadata is None:
            msg = f"Unknown field name: {name}"
            raise ValueError(msg)
        axes: tuple[AxisName, ...] = metadata.get("__axes__", ())
        return axes


@dataclass(slots=True)
class ResultField(AxisMetadata):
    """Base class for result field containers that carry axis metadata.

    See [Working with Results](../optimizer_setup/results.md#axes-and-dimensionality) for
    how axis metadata is used.
    """
