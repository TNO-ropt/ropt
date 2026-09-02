from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ._frame_core import _iter_field_data

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ropt.enums import AxisName

    from ._results import Results


def _to_pandas_frame(
    results: Results,
    field_name: str,
    select: Iterable[str],
    unstack: Iterable[AxisName] | None,
) -> pd.DataFrame:
    if unstack is None:
        unstack = []
    result_field = getattr(results, field_name)
    joined_frame = pd.DataFrame()
    for field_data in _iter_field_data(result_field, select, results.names):
        index: pd.Index[Any]
        if field_data.axes:
            index = pd.MultiIndex.from_product(
                [(results.batch_id,), *field_data.labels],
                names=("batch_id", *(axis.value for axis in field_data.axes)),
            )
        else:
            index = pd.Index([results.batch_id] * field_data.data.size, name="batch_id")
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
