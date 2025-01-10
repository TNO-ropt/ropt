"""Write optimization results to tabular files."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Final, Literal

from ._data_frame import ResultsDataFrame

if TYPE_CHECKING:
    from pathlib import Path


_HAVE_PANDAS: Final = find_spec("pandas") is not None
_HAVE_TABULATE: Final = find_spec("tabulate") is not None

if TYPE_CHECKING:
    from pathlib import Path

if TYPE_CHECKING and _HAVE_PANDAS:
    import pandas as pd  # noqa: TC002

if _HAVE_TABULATE:
    from tabulate import tabulate


class ResultsTable(ResultsDataFrame):
    """Generate files containing tables of optimization results.

    This class derives from the
    [`ResultsDataFrame`][ropt.report.ResultsDataFrame] class and writes the
    generated data frame in a tabular format to a text file.
    """

    def __init__(
        self,
        columns: dict[str, str],
        path: Path,
        *,
        table_type: Literal["functions", "gradients"] = "functions",
        min_header_len: int | None = None,
    ) -> None:
        """Initialize a results table.

        The `columns` parameter specifies which results are to be exported. The
        keys of the `columns` dictionary correspond to the `fields` parameter of
        the [`ResultsDataFrame`][ropt.report.ResultsDataFrame] parent class. The
        values are the corresponding titles of the columns of the table that is
        generated. As described in the documentation of the parent class, a
        single field may generate multiple columns, each with unique names
        (i.e., variable names). These are handled by adding the name to the
        column name below the main title. As a result, the header may consist of
        multiple lines, and the number of lines may vary according to requested
        fields. For a consistent result, the minimum number of header lines can
        be specified via the `min_header_len` argument. When needed, blank lines
        will be added to reach the specified minimum number of header lines.

        Tip: Reading the generated file.
            The resulting table can be read using a reader that can handle
            fixed-width columns, such as the read_fwf function of pandas. However,
            the header will need to skip a number of header lines. The
            min_header_len argument can be used to set the minimum number of lines
            in the header. If the generated header has fewer lines than
            min_header_len, empty lines will be added. For example:

            ```py
            # For a table generated with `min_header_len=3`:
            results = pd.read_fwf(
                "results.txt",
                header=list(range(3)),
                skip_blank_lines=False,
                skiprows=[3],
            )
            ```

        Args:
            columns:        Mapping of column names for the results table.
            path:           Path of the result file.
            table_type:     Type of the table.
            min_header_len: Minimal number of header lines.

        Raises:
            NotImplementedError: If the pandas or tabulate modules are not
                                 available
        """
        if not (_HAVE_TABULATE and _HAVE_PANDAS):
            msg = "ResultsTable requires the `tabulate` and `pandas` modules"
            raise NotImplementedError(msg)

        super().__init__(set(columns), table_type=table_type)

        if path.parent.exists():
            if not path.parent.is_dir():
                msg = f"Cannot write table to: {path}"
                raise RuntimeError(msg)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        self._columns = columns
        self._path = path
        self._min_header_len = min_header_len

    def save(self) -> None:
        """Write the table to a file."""
        frame = self._frame.reset_index()
        table = _extract_columns(frame, mapping=self._columns)
        _write_table(table, self._path, self._min_header_len)


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
