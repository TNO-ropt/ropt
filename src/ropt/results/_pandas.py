from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any, Iterable, cast

import pandas as pd

if TYPE_CHECKING:
    from ropt.enums import ResultAxis

    from ._result_field import ResultField


def _to_dataframe(  # noqa: PLR0913
    result_field: ResultField,
    plan_id: tuple[int, ...],
    result_id: int | tuple[int, ...],
    batch_id: int | None,
    select: Iterable[str] | None,
    unstack: Iterable[ResultAxis] | None,
    names: dict[ResultAxis, tuple[str, ...] | None] | None = None,
) -> pd.DataFrame:
    if select is None:
        select = (field.name for field in fields(result_field))
    if unstack is None:
        unstack = []
    joined_frame = pd.DataFrame()
    for field in select:
        series = _to_series(result_field, plan_id, result_id, batch_id, field, names)
        if series is not None:
            frame = series.to_frame()
            for axis in unstack:
                if axis.value in frame.index.names:
                    frame = cast(pd.DataFrame, frame.unstack(axis.value))  # noqa: PD010
            frame.columns = frame.columns.to_flat_index()  # type:ignore[no-untyped-call]
            if joined_frame.empty:
                joined_frame = frame
            else:
                joined_frame = joined_frame.join(frame, how="inner")
    return joined_frame


def _to_series(  # noqa: PLR0913
    result_field: ResultField,
    plan_id: tuple[int, ...],
    result_id: int | tuple[int, ...],
    batch_id: int | None,
    field: str,
    names: dict[ResultAxis, tuple[str, ...] | None] | None = None,
) -> pd.Series[Any] | None:
    try:
        data = getattr(result_field, field)
    except AttributeError as exc:
        msg = f"Not a field name: {field}"
        raise ValueError(msg) from exc
    if data is None:
        return None
    axes = result_field.get_axes(field)
    if names is None:
        names = {}
    indices = [
        pd.RangeIndex(data.shape[idx]) if index is None else index
        for idx, index in enumerate(names.get(axis, None) for axis in axes)
    ]
    series: pd.Series[Any]
    index: tuple[Any, ...] = (plan_id, result_id, 0 if batch_id is None else batch_id)
    index_names = ["plan_id", "result_id", "batch_id"]
    if indices:
        multi_index = pd.MultiIndex.from_product(
            indices, names=(axis.value for axis in axes)
        )
        series = pd.Series(data.flatten(), index=multi_index, name=field)
        series = pd.concat({index: series}, names=index_names)
    else:
        series = pd.Series(
            data,
            index=pd.MultiIndex.from_tuples([index], names=index_names),
            name=field,
        )
    return series
