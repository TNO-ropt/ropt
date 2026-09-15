"""Backend-neutral helpers for exporting results to data frames.

The helpers in this module extract the numpy data and the axis labels from
result fields, without committing to a specific data frame implementation.
They are shared by the pandas and polars exporters to guarantee that both
produce identical column names, values and row ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn

import numpy as np

from ropt.enums import AxisName

from ._result_field import AxisMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from numpy.typing import NDArray

    from ._results import Results


@dataclass(slots=True)
class FieldData:
    """The data extracted from a single sub-field of a result field."""

    name: str
    """The name of the sub-field, including the key for dict-valued fields."""

    data: NDArray[Any]
    """The values of the sub-field, flattened in C order."""

    axes: tuple[str, ...]
    """The axes of the sub-field, in array order.

    A builtin axis is an [`AxisName`][ropt.enums.AxisName] value; a user axis
    carries the name of the metadata key that defines it.
    """

    labels: tuple[tuple[str | int, ...], ...]
    """The labels of each axis, in the same order as `axes`."""


UNSTACK_AXES: Final[dict[str, tuple[str, ...]]] = {
    "functions": (
        AxisName.OBJECTIVE,
        AxisName.NONLINEAR_CONSTRAINT,
        AxisName.VARIABLE,
        AxisName.LINEAR_CONSTRAINT,
    ),
    "gradients": (
        AxisName.OBJECTIVE,
        AxisName.NONLINEAR_CONSTRAINT,
        AxisName.VARIABLE,
    ),
}
"""The axes that an aggregated frame unstacks into columns.

One list per result type: within a result type an axis is either always
unstacked or never, so `realization` and `perturbation` stay as row labels.
"""


def _reject(path: str, reason: str) -> NoReturn:
    msg = f"{reason}: {path}"
    raise ValueError(msg)


def _resolve_path(
    results: Results, path: str
) -> tuple[AxisMetadata, str, list[str]] | None:
    # Follow attributes while the current object is a result or result field.
    # The first object that is neither ends the walk: that one is the field,
    # and any segments left over address entries inside it.
    segments = path.split(".")
    if not all(segments):
        _reject(path, "Not a correct field name")
    owner: Any = results
    target: Any = results
    for index, segment in enumerate(segments):
        if not isinstance(target, AxisMetadata):
            return owner, segments[index - 1], segments[index:]
        owner = target
        if not hasattr(owner, segment):
            _reject(path, "Not a field name")
        target = getattr(owner, segment)
    if target is None:
        return None
    if isinstance(target, AxisMetadata):
        _reject(path, "Field holds sub-fields, not a value")
    if isinstance(target, Mapping):
        _reject(path, "Field holds a mapping, add a key")
    return owner, segments[-1], []


def _get_field_data(
    results: Results,
    path: str,
    names: dict[str, tuple[str | int, ...]],
) -> FieldData | None:
    resolved = _resolve_path(results, path)
    if resolved is None:
        return None
    owner, field, keys = resolved
    data: Any = getattr(owner, field)
    for key in keys:
        if not isinstance(data, Mapping):
            _reject(path, "Field holds no mapping at this depth")
        if key not in data:
            return None
        data = data[key]
    if data is None:
        return None
    values = np.asarray(data)
    axes = owner.get_axes(field)
    if keys and values.ndim == len(axes) + 1:
        # An array-valued mapping entry spans one extra axis, named after the
        # key that holds it, so its labels can be looked up like any other.
        axes = (*axes, keys[-1])
    labels = tuple(
        tuple(range(values.shape[idx])) if labels is None else tuple(labels)
        for idx, labels in enumerate(names.get(axis) for axis in axes)
    )
    return FieldData(name=path, data=values.flatten(), axes=axes, labels=labels)


def _iter_field_data(
    results: Results,
    select: Iterable[str],
    names: dict[str, tuple[str | int, ...]],
) -> Iterator[FieldData]:
    for path in select:
        field_data = _get_field_data(results, path, names)
        if field_data is not None:
            yield field_data


_KEY_COLUMNS: Final = frozenset(
    {"batch_id", *(axis.value for axis in AxisName)},
)

_BUILTIN_AXES: Final = frozenset(AxisName)


def _unstack_order(
    field_data: FieldData, unstack: Sequence[str], *, aggregated: bool
) -> list[str]:
    wanted = list(unstack)
    if aggregated:
        wanted += [
            axis
            for axis in field_data.axes
            if axis not in _BUILTIN_AXES and axis not in wanted
        ]
    return [axis for axis in wanted if axis in field_data.axes]


def _check_unstack_axes(unstack: Sequence[str], seen: set[str]) -> None:
    # Without this an unrecognised axis is silently dropped, and the caller gets
    # a stacked frame with no indication that the name was never matched.
    unknown = [axis for axis in unstack if axis not in seen]
    if unknown:
        msg = f"Unknown axes to unstack: {', '.join(unknown)}"
        raise ValueError(msg)


def _value_fields(sub_fields: set[str]) -> list[str]:
    # Key columns are emitted by the exporter itself, so a request for one of
    # them names a column to keep, not a field to read. Sorting keeps the
    # column order of an aggregated frame independent of set iteration order.
    return sorted(sub_fields - _KEY_COLUMNS)


def _has_results(
    results: Results, result_type: Literal["functions", "gradients"]
) -> bool:
    # These are None if too few realizations succeeded to aggregate them.
    if result_type == "functions":
        return getattr(results, "functions", None) is not None
    return getattr(results, "gradients", None) is not None
