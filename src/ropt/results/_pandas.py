from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ._frame_core import _iter_field_data

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ropt.enums import AxisName

    from ._result_field import ResultField


def _to_dataframe(
    result_field: ResultField,
    batch_id: int,
    select: Iterable[str],
    unstack: Iterable[AxisName] | None,
    names: dict[str, tuple[str | int, ...]],
) -> pd.DataFrame:
    if unstack is None:
        unstack = []
    joined_frame = pd.DataFrame()
    for field_data in _iter_field_data(result_field, select, names):
        index: pd.Index[Any]
        if field_data.axes:
            index = pd.MultiIndex.from_product(
                [(batch_id,), *field_data.labels],
                names=("batch_id", *(axis.value for axis in field_data.axes)),
            )
        else:
            index = pd.Index([batch_id] * field_data.data.size, name="batch_id")
        frame = pd.DataFrame({field_data.name: field_data.data}, index=index)
        levels = [axis.value for axis in unstack if axis.value in frame.index.names]
        if levels:
            frame = frame.reset_index().pivot_table(
                index=[col for col in frame.index.names if col not in levels],
                columns=levels,
                aggfunc="first",
                sort=False,
            )
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.to_flat_index()
        if joined_frame.empty:
            joined_frame = frame
        else:
            joined_frame = joined_frame.join(frame, how="inner")
    return joined_frame
