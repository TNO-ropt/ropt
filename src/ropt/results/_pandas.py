from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ropt.enums import AxisName

    from ._result_field import ResultField


def _to_dataframe(
    result_field: ResultField,
    batch_id: int | None,
    select: Iterable[str],
    unstack: Iterable[AxisName] | None,
    names: dict[str, tuple[str | int, ...]],
) -> pd.DataFrame:
    if unstack is None:
        unstack = []
    joined_frame = pd.DataFrame()
    for field in select:
        split_field, sep, key = field.partition(".")
        if sep and not key:
            msg = f"Not a correct field name: {field}"
            raise ValueError(msg)
        series = _to_series(result_field, split_field if key else field, key, names)
        if series is not None:
            frame = pd.concat(
                {(0 if batch_id is None else batch_id): series}, names=["batch_id"]
            ).to_frame()
            levels = [axis.value for axis in unstack if axis.value in frame.index.names]
            if levels:
                frame = frame.reset_index().pivot_table(
                    index=[col for col in frame.index.names if col not in levels],
                    columns=levels,
                    aggfunc="first",
                    sort=False,
                )
            frame.columns = frame.columns.to_flat_index()
            if joined_frame.empty:
                joined_frame = frame
            else:
                joined_frame = joined_frame.join(frame, how="inner")
    if not joined_frame.empty and "_dummy_" in joined_frame.index.names:
        return joined_frame.droplevel("_dummy_")
    return joined_frame


def _to_series(
    result_field: ResultField,
    field: str,
    key: str | None,
    names: dict[str, tuple[str | int, ...]],
) -> pd.Series[Any] | None:
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
    indices = [
        list(range(data.shape[idx])) if index is None else index
        for idx, index in enumerate(names.get(axis) for axis in axes)
    ]
    series: pd.Series[Any]
    if indices:
        multi_index = pd.MultiIndex.from_product(
            indices, names=tuple(axis.value for axis in axes)
        )
        series = pd.Series(
            data.flatten(), index=multi_index, name=f"{field}.{key}" if key else field
        )
    else:
        series = pd.Series(data, name=f"{field}.{key}" if key else field)
        series.index.name = "_dummy_"

    return series
