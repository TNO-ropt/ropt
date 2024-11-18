from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ropt.enums import ResultAxisName


TypeResultField = TypeVar("TypeResultField", bound="ResultField")


@dataclass(slots=True)
class ResultField:
    """Base class for `Results` fields."""

    @classmethod
    def get_axis_names(cls, name: str) -> tuple[ResultAxisName, ...]:
        """Return the axis names of a field in the given field class or object.

        When used with the class or an instance of that class for one of the
        fields of a `Results` object, retrieve the names of the axes of the
        stored `numpy` array from the metadata and return them.

        Args:
            name: The name of the sub-field in the instance or class.

        Raises:
            ValueError: Raised if an unknown field name is passed.

        Returns:
            A tuple listing the names of the axes.
        """
        metadata = next(
            (item.metadata for item in fields(cls) if item.name == name), None
        )
        if metadata is None:
            msg = f"Unknown field name: {name}"
            raise ValueError(msg)
        return metadata.get("__axes__", ())
