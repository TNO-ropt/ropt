from __future__ import annotations

from dataclasses import fields
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from pathlib import Path

    from ropt.results import Results

_HAVE_PANDAS: Final = find_spec("pandas") is not None
_HAVE_TABULATE: Final = find_spec("tabulate") is not None

if TYPE_CHECKING and _HAVE_PANDAS:
    import pandas as pd  # noqa: TC002

if _HAVE_TABULATE:
    from tabulate import tabulate


def _get_select(results: Results, field_name: str, sub_fields: set[str]) -> list[str]:
    results_field = getattr(results, field_name)
    return [
        name
        for name in {field.name for field in fields(results_field)}
        if f"{field_name}.{name}" in sub_fields
    ]


def _add_prefix(name: tuple[str, ...] | str, prefix: str) -> tuple[str, ...] | str:
    return (
        (f"{prefix}.{name[0]}",) + name[1:]
        if isinstance(name, tuple)
        else f"{prefix}.{name}"
    )


def _add_metadata(
    data_frame: pd.DataFrame, results: Results, sub_fields: set[str]
) -> pd.DataFrame:
    for field in sub_fields:
        split_fields = field.split(".")
        if split_fields[0] == "metadata":
            value = _get_value(results.metadata, split_fields[1:])
            if value is not None:
                data_frame[field] = value
    return data_frame


def _get_value(data: dict[str, Any], keys: list[str]) -> Any | None:  # noqa: ANN401
    for key in keys:
        if isinstance(data, dict):
            if key not in data:
                return None
            data = data[key]
        else:
            break
    return data


def _write_table(table: pd.DataFrame, path: Path, min_header_len: int | None) -> None:
    if not table.empty:
        data = _align_column_names(table, min_header_len)
        table_data = {str(column): data[column] for column in data}
        path.write_text(
            tabulate(table_data, headers="keys", tablefmt="simple", showindex=False),
        )


def _extract_columns(data_frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    # Columns need to be reordered to follow the ordering of the mapping keys.
    # Column names may be tuples, where the first element indicates column type
    # (e.g. `variable` or `objective`) and the rest variable or function names.
    # Hence, sets of columns exist where the first element of the column name
    # tuple is identical. Re-ordering is done according to the first element of
    # these sets of columns, ordering within the set is unchanged.
    reordered_columns = [
        name
        for key in mapping
        for name in data_frame.columns.to_numpy()
        if name == key or (isinstance(name, tuple) and name[0] == key)
    ]
    data_frame = data_frame.reindex(columns=reordered_columns)

    # Columns are renamed according to the mapping. If the column name is a
    # tuple the first element is renamed. Tuples are joined by newlines, as a
    # result column type (e.g. `variable` or `objective`) will be on the first
    # line, whereas the name of the variable or the function will be on the
    # second. Columns names that are not a tuple may contain new lines in the
    # mapped file, to split long names over multiple lines.
    renamed_columns = [
        "\n".join([mapping[name[0]]] + [str(item) for item in name[1:]])
        if isinstance(name, tuple)
        else mapping[name]
        for name in reordered_columns
    ]
    return data_frame.set_axis(renamed_columns, axis="columns")


def _align_column_names(
    data_frame: pd.DataFrame, min_header_len: int | None
) -> pd.DataFrame:
    max_lines = max(len(str(column).split("\n")) for column in data_frame.columns)
    if min_header_len is not None and max_lines < min_header_len:
        max_lines = min_header_len
    return data_frame.rename(
        columns={
            column: str(column) + (max_lines - len(str(column).split("\n"))) * "\n"
            for column in data_frame.columns
        },
    )
