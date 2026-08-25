from __future__ import annotations

import threading
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from ropt.enums import EnOptEventType
from ropt.exceptions import UnsupportedError
from ropt.results import (
    DomainType,
    Results,
    results_to_dataframe,
    results_to_polars,
)

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from ropt.events import EnOptEvent

_HAVE_PANDAS: Final = find_spec("pandas") is not None
_HAVE_POLARS: Final = find_spec("polars") is not None

if _HAVE_PANDAS:
    import pandas as pd

if _HAVE_POLARS:
    import polars as pl

Backend = Literal["pandas", "polars"]


_FUNCTION_TABLES: Final[dict[str, dict[str, str]]] = {
    "functions": {
        "batch_id": "Batch",
        "functions.target_objective": "Total-Objective",
        "functions.objectives": "Objective",
        "functions.constraints": "Constraint",
        "evaluations.variables": "Variable",
    },
    "evaluations": {
        "batch_id": "Batch",
        "realization": "Realization",
        "variable": "Variable-name",
        "evaluations.variables": "Variable",
        "evaluations.objectives": "Objective",
        "evaluations.constraints": "Constraint",
    },
    "constraints": {
        "batch_id": "Batch",
        "constraint_info.bound_lower": "BCD-lower",
        "constraint_info.bound_upper": "BCD-upper",
        "constraint_info.linear_lower": "ICD-lower",
        "constraint_info.linear_upper": "ICD-upper",
        "constraint_info.nonlinear_lower": "OCD-lower",
        "constraint_info.nonlinear_upper": "OCD-upper",
        "constraint_info.bound_violation": "BCD-violation",
        "constraint_info.linear_violation": "ICD-violation",
        "constraint_info.nonlinear_violation": "OCD-violation",
    },
}
_GRADIENT_TABLES: Final[dict[str, dict[str, str]]] = {
    "gradients": {
        "batch_id": "Batch",
        "gradients.target_objective": "Total-Gradient",
        "gradients.objectives": "Grad-objective",
        "gradients.constraints": "Grad-constraint",
    },
    "perturbations": {
        "batch_id": "Batch",
        "realization": "Realization",
        "perturbation": "Perturbation",
        "evaluations.perturbed_variables": "Variable",
        "evaluations.perturbed_objectives": "Objective",
        "evaluations.perturbed_constraints": "Constraint",
    },
}


class DataFrameHandler(EventHandler):
    """Build pandas or polars DataFrames from optimization results.

    Collects [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults] into named tables.
    Tables are defined via `add_table` with a column specification, or
    registered in bulk with `set_default_tables`.

    The frame library is chosen with the `backend` argument. The `"pandas"`
    backend produces tables indexed by batch and axis labels, the `"polars"`
    backend produces long-format tables in which those become ordinary leading
    columns.

    Access tables via dictionary syntax: `handler["functions"]`.

    Warning:
        Tables are generated on the fly from internal data when accessing them
        in this way. When multiple accesses are needed, it is more efficient to
        first store them in a variable.

    See [The Simple API](../running/running.md#dataframehandler) for the column
    specification, default tables, and callback details.
    """

    def __init__(
        self,
        *,
        backend: Backend = "pandas",
        sep: str = ",",
    ) -> None:
        """Initialize a default table event handler.

        Args:
            backend:   Frame library used to build the tables.
            sep:       Separator used in column names.

        Raises:
            ValueError:       If `backend` is not `"pandas"` or `"polars"`.
            UnsupportedError: If the selected backend module is not installed.
        """
        if backend not in {"pandas", "polars"}:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)
        if backend == "pandas" and not _HAVE_PANDAS:
            msg = "DataFrameHandler requires the pandas module; install ropt[pandas]."
            raise UnsupportedError(msg)
        if backend == "polars" and not _HAVE_POLARS:
            msg = "DataFrameHandler requires the polars module; install ropt[polars]."
            raise UnsupportedError(msg)

        super().__init__()
        self._backend: Backend = backend
        self._sep = sep
        self._callback: Callable[[Path | None], None] | None = None
        self._tables: dict[str, _ResultsTable] = {}

    @property
    def backend(self) -> Backend:
        """The frame library used to build the tables.

        Returns:
            Either `"pandas"` or `"polars"`.
        """
        return self._backend

    def set_default_tables(self, *, domain: DomainType = "user") -> None:
        """Register a standard set of result tables.

        Adds the default `functions`, `evaluations`, and `constraints` tables
        for function results, and the default `gradients` and `perturbations`
        tables for gradient results.

        Args:
            domain: Domain (`"user"` or `"optimizer"`) the tables are filled
                from. The `"user"` domain reports values as seen by the user;
                `"optimizer"` reports them in the optimizer's transformed space.
        """
        for name, columns in _FUNCTION_TABLES.items():
            self.add_table(name, "functions", columns, domain=domain)
        for name, columns in _GRADIENT_TABLES.items():
            self.add_table(name, "gradients", columns, domain=domain)

    def set_callback(self, callback: Callable[[Path | None], None]) -> None:
        """Set a function to call whenever the tables are updated.

        The callback receives the output directory configured for the run
        ([`OptimizerConfig.output_dir`][ropt.config.OptimizerConfig], `None` if
        it is not set), and reads the tables from this handler. If it performs
        blocking operations (for example writing tables to disk), register this
        handler with `run_in_thread=True` on the
        [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher]:

        ```python
        event_dispatcher.add_event_handler(table_handler, run_in_thread=True)
        ```

        Args:
            callback: A function that is called when the tables are updated.
        """
        self._callback = callback

    def add_table(
        self,
        name: str,
        table_type: Literal["functions", "gradients"],
        columns: dict[str, str],
        domain: DomainType = "user",
    ) -> None:
        """Register a new table to be populated from incoming results.

        Args:
            name:       Key under which the table is stored and looked up.
            table_type: Whether this table is filled from function results
                        (`"functions"`) or gradient results (`"gradients"`).
            columns:    Mapping from result-field attribute names (using dotted
                        attribute syntax) to display titles.
            domain:     Domain (`"user"` or `"optimizer"`) the table is filled
                        from.
        """
        self._tables[name] = _ResultsTable(
            columns,
            table_type=table_type,
            domain=domain,
            backend=self._backend,
            sep=self._sep,
        )

    def get_tables(self) -> dict[str, pd.DataFrame | pl.DataFrame]:
        """Return the tables stored in the event handler.

        Returns:
            A dictionary mapping table names to their corresponding tables.

        Warning:
            Tables are generated on the fly from internal data. When multiple
            access is needed, it is more efficient to first store them in a
            variable.
        """
        return {key: table.get_table() for key, table in self._tables.items()}

    def handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Args:
            event: The event object.
        """
        results = event.results
        if results:
            transformed_results = (
                tuple(item.transform_from_optimizer(event.context) for item in results)
                if any(table.domain == "user" for table in self._tables.values())
                else ()
            )
            done = [
                table.add_results(transformed_results)
                if table.domain == "user"
                else table.add_results(results)
                for table in self._tables.values()
            ]
            if any(done) and self._callback is not None:
                self._callback(event.context.optimizer.output_dir)

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EnOptEventType.FINISHED_EVALUATION}

    def __getitem__(self, key: str) -> Any:  # ruff: ignore[any-type]
        """Retrieve a of a table from the event handler.

        Warning:
            The table is generated on the fly from internal data hen multiple
            access are needed, it is more efficient to first store them in a
            variable.

        Args:
            key: The string key identifying the table to retrieve.

        Returns:
            The table associated with the specified key.

        Raises:
            AttributeError: If the requested table does not exist.
        """
        if key not in self._tables:
            msg = f"Unknown table: `{key}`"
            raise AttributeError(msg)
        return self._tables[key].get_table()

    def add_column(self, table: str, name: str, title: str) -> None:
        """Add a column to a given table.

        Args:
            table: The name of the table to add the column to.
            name:  The name of the field to add as a column, using attribute syntax.
            title: The title of the column to add.
        """
        self._tables[table].add_column(name, title)


class _ResultsTable:
    # One frame is built per batch of results and only concatenated on access,
    # so a long run does not rebuild a growing table on every event. The lock is
    # for the handler running behind a dispatcher thread while the tables are
    # read from another.
    def __init__(
        self,
        columns: dict[str, str],
        table_type: Literal["functions", "gradients"],
        *,
        domain: DomainType = "user",
        backend: Backend = "pandas",
        sep: str = ",",
    ) -> None:
        self._columns = columns
        self._results_type = table_type
        self._domain = domain
        self._backend = backend
        self._sep = sep
        self._frames: list[pd.DataFrame | pl.DataFrame] = []
        self._lock = threading.Lock()

    @property
    def domain(self) -> DomainType:
        return self._domain

    def add_column(self, name: str, title: str) -> None:
        with self._lock:
            self._columns[name] = title

    def add_results(self, results: Sequence[Results]) -> bool:
        with self._lock:
            columns = set(self._columns)
        if self._backend == "polars":
            polars_frame = results_to_polars(
                results, columns, result_type=self._results_type, sep=self._sep
            )
            if polars_frame.height == 0:
                return False
            frame: pd.DataFrame | pl.DataFrame = polars_frame
        else:
            pandas_frame = results_to_dataframe(
                results, columns, result_type=self._results_type
            )
            if pandas_frame.empty:
                return False
            frame = pandas_frame
        with self._lock:
            self._frames.append(frame)
        return True

    def get_table(self) -> pd.DataFrame | pl.DataFrame:
        with self._lock:
            frames = list(self._frames)
            columns = dict(self._columns)
        if self._backend == "polars":
            if not frames:
                return pl.DataFrame()
            return _build_polars_table(
                cast("list[pl.DataFrame]", frames), columns, self._sep
            )
        if not frames:
            return pd.DataFrame()
        return _build_pandas_table(
            cast("list[pd.DataFrame]", frames), columns, self._sep
        )


def _build_pandas_table(
    frames: list[pd.DataFrame], columns: dict[str, str], sep: str
) -> pd.DataFrame:
    data = pd.concat(frames).reset_index()
    reordered_columns = [
        name
        for key in columns
        for name in data.columns.to_numpy()
        if name == key or (isinstance(name, tuple) and name[0] == key)
    ]
    data = data.reindex(columns=reordered_columns)
    renamed_columns = [
        (str(columns[name[0]]), *(str(item) for item in name[1:]))
        if isinstance(name, tuple)
        else columns[name]
        for name in data.columns.to_numpy()
    ]
    data.columns = [
        sep.join(item) if isinstance(item, tuple) else item for item in renamed_columns
    ]
    return data


def _build_polars_table(
    frames: list[pl.DataFrame], columns: dict[str, str], sep: str
) -> pl.DataFrame:
    data = pl.concat(frames, how="diagonal")
    reordered_columns: list[str] = []
    renamed_columns: dict[str, str] = {}
    for key, title in columns.items():
        for name in data.columns:
            if name == key:
                reordered_columns.append(name)
                renamed_columns[name] = title
            elif name.startswith(key + sep):
                reordered_columns.append(name)
                renamed_columns[name] = title + name[len(key) :]
    return data.select(reordered_columns).rename(renamed_columns)
