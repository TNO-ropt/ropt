"""Write optimization results to tabular files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple

from ._data_frame import ResultsDataFrame
from ._utils import _HAVE_PANDAS, _HAVE_TABULATE, _extract_columns, _write_table

if TYPE_CHECKING:
    from pathlib import Path

    from ropt.config.enopt import EnOptConfig
    from ropt.results import Results

    if _HAVE_PANDAS:
        import pandas as pd


class ResultsTable(ResultsDataFrame):
    """Generate files containing tables of optimization results.

    This class derives from the
    [`ResultsDataFrame`][ropt.report.ResultsDataFrame] class and writes the
    generated data frame in a tabular format to a text file.
    """

    def __init__(  # noqa: PLR0913
        self,
        columns: Dict[str, str],
        path: Path,
        *,
        filters: Optional[
            Dict[str, Callable[[pd.Series[Any]], pd.Series[bool]]]
        ] = None,
        table_type: Literal["functions", "gradients"] = "functions",
        min_header_len: Optional[int] = None,
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

        The optional `filters` argument can be used to pass functions that can
        be used to filter the rows before generating the table. The keys of the
        `filters` dictionary should be the columns to apply the filter function
        to, and the values the filter functions, which should expect a pandas
        series and transform those into an equally sized series of boolean
        values. For example, to retain only realizations with the name equal to
        2:

        ```python filters = {"realization": lambda x: x == 2}
        ```

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
            columns:        Mapping of column names for the results table
            path:           Optional location of the result file
            filters:        Optional filter functions to remove rows
            table_type:     Type of the table
            min_header_len: Minimal number of header lines

        Raises:
            NotImplementedError: If the pandas or tabulate modules are not
                                 available
        """
        if not (_HAVE_TABULATE and _HAVE_PANDAS):
            msg = "ResultsTable requires the `tabulate` and `pandas` modules"
            raise NotImplementedError(msg)

        super().__init__(set(columns), table_type=table_type)
        self._columns = columns
        self._path = path
        self._filters = filters
        self._min_header_len = min_header_len

    def add_results(self, config: EnOptConfig, results: Tuple[Results, ...]) -> bool:
        """Add results to the table.

        This method can be called directly from any observers connected to
        events that produce results (see
        [`add_observer`][ropt.optimization.EnsembleOptimizer.add_observer]).

        Args:
            config:  The configuration of the optimizer generating the results
            results: The results to add

        Returns:
            True if a result was added, else False
        """
        if super().add_results(config, results):
            frame = self._frame.reset_index()
            if self._filters is not None:
                for key, filter_ in self._filters.items():
                    if key in frame.columns:
                        frame = frame[filter_(frame[key])]
            table = _extract_columns(frame, mapping=self._columns)
            _write_table(table, self._path, self._min_header_len)
            return True
        return False
