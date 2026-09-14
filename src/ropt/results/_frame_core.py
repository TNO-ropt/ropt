"""Backend-neutral helpers for exporting results to data frames.

The helpers in this module extract the numpy data and the axis labels from
result fields, without committing to a specific data frame implementation.
They are shared by the pandas and polars exporters to guarantee that both
produce identical column names, values and row ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from ropt.enums import AxisName

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

    from ._result_field import AxisMetadata
    from ._results import Results


@dataclass(slots=True)
class FieldData:
    """The data extracted from a single sub-field of a result field."""

    name: str
    """The name of the sub-field, including the key for dict-valued fields."""

    data: NDArray[Any]
    """The values of the sub-field, flattened in C order."""

    axes: tuple[AxisName, ...]
    """The axes of the sub-field, in array order."""

    labels: tuple[tuple[str | int, ...], ...]
    """The labels of each axis, in the same order as `axes`."""


@dataclass(frozen=True, slots=True)
class FrameSpec:
    """The specification of one result field of an aggregated frame."""

    field: str
    """The name of the result field to export."""

    unstack: tuple[AxisName, ...]
    """The axes of the field that are unstacked into columns."""

    has_sub_fields: bool = True
    """Whether the field holds sub-fields, rather than being a value itself."""


_VARIABLE_AXES: Final = (AxisName.VARIABLE,)
_VALUE_AXES: Final = (AxisName.OBJECTIVE, AxisName.NONLINEAR_CONSTRAINT)
_GRADIENT_AXES: Final = (
    AxisName.OBJECTIVE,
    AxisName.NONLINEAR_CONSTRAINT,
    AxisName.VARIABLE,
)
_CONSTRAINT_AXES: Final = (
    AxisName.VARIABLE,
    AxisName.LINEAR_CONSTRAINT,
    AxisName.NONLINEAR_CONSTRAINT,
)

FRAME_SPECS: Final[dict[str, tuple[FrameSpec, ...]]] = {
    "functions": (
        FrameSpec("functions", _VALUE_AXES),
        FrameSpec("scaled.functions", _VALUE_AXES),
        FrameSpec("target_objective", (), has_sub_fields=False),
        FrameSpec("variables", _VARIABLE_AXES, has_sub_fields=False),
        FrameSpec("scaled.variables", _VARIABLE_AXES, has_sub_fields=False),
        FrameSpec("evaluations", _VALUE_AXES),
        FrameSpec("constraint_info", _CONSTRAINT_AXES),
        FrameSpec("scaled.constraint_info", _CONSTRAINT_AXES),
    ),
    "gradients": (
        FrameSpec("gradients", _GRADIENT_AXES),
        FrameSpec("scaled.gradients", _GRADIENT_AXES),
        FrameSpec("target_gradient", _VARIABLE_AXES, has_sub_fields=False),
        FrameSpec("variables", _VARIABLE_AXES, has_sub_fields=False),
        FrameSpec("scaled.variables", _VARIABLE_AXES, has_sub_fields=False),
        FrameSpec("perturbed_variables", _VARIABLE_AXES, has_sub_fields=False),
        FrameSpec("scaled.perturbed_variables", _VARIABLE_AXES, has_sub_fields=False),
        FrameSpec("evaluations", _VALUE_AXES),
    ),
}
"""The fields that make up an aggregated frame, and the axes they unstack.

Fields that are `None` on a given result are skipped.
"""


def _resolve_field(results: Results, field_name: str) -> Any | None:  # ruff: ignore[any-type]
    target: Any = results
    for part in field_name.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def _get_field_data(
    result_field: AxisMetadata,
    name: str,
    names: dict[str, tuple[str | int, ...]],
) -> FieldData | None:
    field, separator, key = name.partition(".")
    if separator and not key:
        msg = f"Not a correct field name: {name}"
        raise ValueError(msg)
    try:
        data = getattr(result_field, field)
    except AttributeError as exc:
        msg = f"Not a field name: {field}"
        raise ValueError(msg) from exc
    if data is None:
        return None
    if key:
        if key not in data:
            return None
        data = data[key]
    axes = result_field.get_axes(field)
    labels = tuple(
        tuple(range(data.shape[idx])) if labels is None else tuple(labels)
        for idx, labels in enumerate(names.get(axis) for axis in axes)
    )
    return FieldData(
        name=f"{field}.{key}" if key else field,
        data=data.flatten(),
        axes=axes,
        labels=labels,
    )


def _iter_spec_data(
    results: Results,
    field_name: str,
    select: Iterable[str],
    names: dict[str, tuple[str | int, ...]],
    *,
    has_sub_fields: bool,
) -> Iterator[FieldData]:
    if has_sub_fields:
        owner = _resolve_field(results, field_name)
        assert owner is not None
        yield from _iter_field_data(owner, select, names)
        return
    # The owner of a value field is the result itself, unless the field is
    # nested, as the scaled fields are.
    owner_name, _, name = field_name.rpartition(".")
    owner = results if not owner_name else _resolve_field(results, owner_name)
    if owner is not None:
        field_data = _get_field_data(owner, name, names)
        if field_data is not None:
            yield field_data


def _iter_field_data(
    result_field: AxisMetadata,
    select: Iterable[str],
    names: dict[str, tuple[str | int, ...]],
) -> Iterator[FieldData]:
    for name in select:
        field_data = _get_field_data(result_field, name, names)
        if field_data is not None:
            yield field_data


def _get_select(field_name: str, sub_fields: set[str]) -> list[str]:
    return [
        item.removeprefix(f"{field_name}.")
        for item in sub_fields
        if item.startswith(f"{field_name}.")
    ]


def _get_value(data: dict[str, Any], keys: list[str]) -> Any | None:  # ruff: ignore[any-type]
    for key in keys:
        if isinstance(data, dict):
            if key not in data:
                return None
            data = data[key]
        else:
            break
    return data


def _has_results(
    results: Results, result_type: Literal["functions", "gradients"]
) -> bool:
    # These are None if too few realizations succeeded to aggregate them.
    if result_type == "functions":
        return getattr(results, "functions", None) is not None
    return getattr(results, "gradients", None) is not None
