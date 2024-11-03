from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple, Union, cast

import pandas as pd

from ropt.enums import ResultAxisName

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig

    from ._result_field import ResultField


def _to_dataframe(  # noqa: PLR0913
    config: EnOptConfig,
    result_field: ResultField,
    result_id: Union[int, Tuple[int, ...]],
    batch_id: Optional[int],
    select: Optional[Iterable[str]],
    unstack: Optional[Iterable[ResultAxisName]],
) -> pd.DataFrame:
    if select is None:
        select = (field.name for field in fields(result_field))
    if unstack is None:
        unstack = []
    joined_frame = pd.DataFrame()
    for field in select:
        series = _to_series(config, result_field, result_id, batch_id, field)
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


def _to_series(
    config: EnOptConfig,
    result_field: ResultField,
    result_id: Union[int, Tuple[int, ...]],
    batch_id: Optional[int],
    field: str,
) -> Optional[pd.Series[Any]]:
    try:
        data = getattr(result_field, field)
    except AttributeError as exc:
        msg = f"Not a field name: {field}"
        raise ValueError(msg) from exc
    if data is None:
        return None
    axes = result_field.get_axis_names(field)
    indices = [_get_index(config, axis) for axis in axes]
    indices = [
        index if index else pd.RangeIndex(data.shape[idx])
        for idx, index in enumerate(indices)
    ]
    series: pd.Series[Any]
    index: Tuple[Any, ...] = (result_id, 0 if batch_id is None else batch_id)
    index_names = ["result_id", "batch_id"]
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


def _get_index(config: EnOptConfig, axis: ResultAxisName) -> Iterable[Any]:
    result: Iterable[Any] = []
    if axis == ResultAxisName.VARIABLE:
        formatted_names = config.variables.get_formatted_names()
        result = [] if formatted_names is None else formatted_names
    elif axis == ResultAxisName.OBJECTIVE:
        result = (
            config.objective_functions.names
            if config.objective_functions.names is not None
            else []
        )
    elif axis == ResultAxisName.NONLINEAR_CONSTRAINT:
        assert config.nonlinear_constraints is not None
        result = (
            config.nonlinear_constraints.names
            if config.nonlinear_constraints.names is not None
            else []
        )
    elif axis == ResultAxisName.REALIZATION:
        result = (
            config.realizations.names if config.realizations.names is not None else []
        )
    return result
