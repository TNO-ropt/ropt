from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._frame_core import _iter_field_data

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ropt.enums import AxisName

    from ._frame_core import FieldData
    from ._results import Results


def _to_polars_frame(
    results: Results,
    field_name: str,
    select: Iterable[str],
    unstack: Iterable[AxisName] | None,
    sep: str,
) -> tuple[pl.DataFrame, list[str]]:
    if unstack is None:
        unstack = []
    result_field = getattr(results, field_name)
    joined_frame: pl.DataFrame | None = None
    keys: list[str] = []
    values: list[str] = []
    for field_data in _iter_field_data(result_field, select, results.names):
        frame = _build_frame(field_data, results.batch_id)
        index = [column for column in frame.columns if column != field_data.name]
        label_order = [axis.value for axis in unstack if axis.value in index]
        if label_order:
            frame = _unstack(frame, field_data, index, label_order, sep)
            index = [column for column in index if column not in label_order]
        if joined_frame is None:
            joined_frame = frame
        else:
            joined_frame = joined_frame.join(
                frame,
                on=[column for column in index if column in keys],
                how="inner",
                maintain_order="left_right",
            )
        keys += [column for column in index if column not in keys]
        values += [column for column in frame.columns if column not in index]
    if joined_frame is None:
        return pl.DataFrame(), keys
    return joined_frame.select(*keys, *values), keys


def _build_frame(field_data: FieldData, batch_id: int) -> pl.DataFrame:
    columns: dict[str, Any] = {
        "batch_id": np.full(field_data.data.size, batch_id, dtype=np.int64)
    }
    if field_data.axes:
        grids = np.meshgrid(
            *(np.asarray(labels) for labels in field_data.labels), indexing="ij"
        )
        for axis, grid in zip(field_data.axes, grids, strict=True):
            columns[axis.value] = grid.ravel()
    columns[field_data.name] = field_data.data
    return pl.DataFrame(columns)


def _unstack(
    frame: pl.DataFrame,
    field_data: FieldData,
    index: Sequence[str],
    label_order: Sequence[str],
    sep: str,
) -> pl.DataFrame:
    index_columns = [column for column in index if column not in label_order]
    unstacked = frame.with_columns(
        pl.concat_str(
            [pl.col(axis).cast(pl.String) for axis in label_order], separator=sep
        ).alias("_labels_")
    ).pivot(
        on="_labels_",
        index=index_columns,
        values=field_data.name,
        aggregate_function="first",
        maintain_order=True,
    )
    return unstacked.rename(
        {
            column: f"{field_data.name}{sep}{column}"
            for column in unstacked.columns
            if column not in index_columns
        }
    )
