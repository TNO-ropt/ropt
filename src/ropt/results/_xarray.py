from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import xarray

from ropt.enums import ResultAxisName

if TYPE_CHECKING:
    from pathlib import Path

    from ropt.config.enopt import EnOptConfig

    from ._result_field import ResultField
    from ._results import Results, TypeResults


def _to_dataset(  # noqa: PLR0913
    config: EnOptConfig,
    result_field: ResultField,
    plan_id: Tuple[int, ...],
    result_id: Union[int, Tuple[int, ...]],
    batch_id: Optional[int],
    metadata: Dict[str, Any],
    select: Optional[Iterable[str]],
) -> xarray.Dataset:
    if select is None:
        select = (field.name for field in fields(result_field))
    arrays = {field: _to_data_array(config, result_field, field) for field in select}
    arrays = {field: array for field, array in arrays.items() if array is not None}
    attrs: Dict[str, Any] = {}
    attrs["plan_id"] = plan_id
    attrs["result_id"] = result_id
    if batch_id is not None:
        attrs["batch_id"] = batch_id
    if metadata:
        attrs["metadata"] = metadata
    return xarray.Dataset(arrays, attrs=attrs)


def _to_data_array(
    config: EnOptConfig, result_field: ResultField, field: str
) -> Optional[xarray.DataArray]:
    try:
        data = getattr(result_field, field)
    except AttributeError as exc:
        msg = f"Not a field name: {field}"
        raise AttributeError(msg) from exc
    if data is None:
        return None
    axes = result_field.get_axis_names(field)
    indices = [_get_index(config, axis) for axis in axes]
    return xarray.DataArray(
        data,
        coords={
            axis.value: list(index)
            for axis, index in zip(axes, indices, strict=False)
            if index is not None
        },
        dims=[axis.value for axis in axes],
    )


def _get_index(config: EnOptConfig, axis: ResultAxisName) -> Optional[Tuple[Any, ...]]:
    if axis == ResultAxisName.VARIABLE:
        return (
            None
            if config.variables.names is None
            else config.variables.get_formatted_names()
        )
    if axis == ResultAxisName.OBJECTIVE:
        return config.objective_functions.names
    if axis == ResultAxisName.NONLINEAR_CONSTRAINT:
        assert config.nonlinear_constraints is not None
        return config.nonlinear_constraints.names
    if axis == ResultAxisName.REALIZATION:
        return config.realizations.names
    return None


def _from_dataset(dataset: xarray.Dataset) -> Dict[str, Any]:
    return {str(name): data.to_numpy() for name, data in dataset.items()}


def _to_netcdf(results: Results, config: EnOptConfig, filename: Path) -> None:
    mode: Literal["w", "a"] = "w"
    metadata: Dict[str, Any] = {
        "plan_id": json.dumps(results.plan_id),
        "result_id": json.dumps(results.result_id),
        "metadata": json.dumps(results.metadata),
    }
    if results.batch_id is not None:
        metadata["batch_id"] = json.dumps(results.batch_id)
    for result_field in fields(results):
        if is_dataclass(getattr(results, result_field.name)):
            dataset = results.to_dataset(config, result_field.name)
            dataset.to_netcdf(filename, mode=mode, group=result_field.name)
            mode = "a"
    if filename.exists():
        xarray.Dataset(attrs=metadata).to_netcdf(
            filename, mode="a", group="__metadata__"
        )


def _from_netcdf(filename: Path, result_type: Type[TypeResults]) -> Dict[str, Any]:
    if filename.suffix != ".nc":
        filename = filename.with_suffix(".nc")
    result: Dict[str, Any] = xarray.open_dataset(filename, group="__metadata__").attrs
    result = {key: json.loads(value) for key, value in result.items()}
    if "plan_id" in result:
        result["plan_id"] = tuple(result["plan_id"])
    for result_field in fields(result_type):
        # Load all fields, skipping those already loaded from the metadata:
        if result_field.name not in result:
            with suppress(OSError):  # Not all fields have to be present
                result[result_field.name] = xarray.open_dataset(
                    filename, group=result_field.name
                )
    return result
