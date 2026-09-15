from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ._frame_core import _check_unstack_axes, _iter_field_data, _unstack_order

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._results import Results


def _to_pandas_frame(
    results: Results,
    select: Iterable[str],
    unstack: Iterable[str] | None,
    *,
    aggregated: bool = False,
) -> pd.DataFrame:
    # An aggregated frame applies one default axis set to every field, so an
    # axis that no field has is filtered rather than reported.
    requested = [] if unstack is None else [str(axis) for axis in unstack]
    joined_frame = pd.DataFrame()
    seen: set[str] = set()
    for field_data in _iter_field_data(results, select, results.names):
        seen.update(field_data.axes)
        index: pd.Index[Any]
        if field_data.axes:
            index = pd.MultiIndex.from_product(
                [(results.batch_id,), *field_data.labels],
                names=("batch_id", *(str(axis) for axis in field_data.axes)),
            )
        else:
            index = pd.Index([results.batch_id] * field_data.data.size, name="batch_id")
        frame = pd.DataFrame({field_data.name: field_data.data}, index=index)
        levels = _unstack_order(field_data, requested, aggregated=aggregated)
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
    if seen and not aggregated:
        _check_unstack_axes(requested, seen)
    return joined_frame
