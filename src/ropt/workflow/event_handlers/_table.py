from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final, Literal

from ropt.enums import EnOptEventType
from ropt.results import DomainType, Results, results_to_dataframe

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.events import EnOptEvent

_HAVE_PANDAS: Final = find_spec("pandas") is not None

if _HAVE_PANDAS:
    import pandas as pd


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


class Table(EventHandler):
    """Build pandas DataFrames from optimization results.

    Collects [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults] into named tables.
    Tables are defined via `add_table` with a column specification, or
    registered in bulk with `set_default_tables`.

    Access tables via dictionary syntax: `handler["functions"]`.

    Warning:
        Tables are generated on the fly from internal data when accessing them
        in this way. When multiple accesses are needed, it is more efficient to
        first store them in a variable.

    See [Optimization Workflows](../usage/workflows.md#table) for full details
    on column specification format, default tables, and callback functionality.
    """

    def __init__(
        self,
        *,
        sep: str = ",",
    ) -> None:
        """Initialize a default table event handler.

        Args:
            sep:       Separator used in column names.
        """
        if not _HAVE_PANDAS:
            msg = "The pandas module must be installed to use Table"
            raise NotImplementedError(msg)

        super().__init__()
        self._sep = sep
        self._callback: Callable[[EnOptEvent], None] | None = None
        self._tables: dict[str, _ResultsTable] = {}

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

    def set_callback(self, callback: Callable[[EnOptEvent], None]) -> None:
        """Set the callback function.

        This callback will called anytime the tables are updated, passing the
        event that caused the tables to be updated.

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
        )

    def get_tables(self) -> dict[str, pd.DataFrame]:
        """Return the tables stored in the event handler.

        Returns:
            A dictionary mapping table names to their corresponding tables.

        Warning:
            Tables are generated on the fly from internal data. When multiple
            access is needed, it is more efficient to first store them in a
            variable.
        """
        return {key: table.get_table(self._sep) for key, table in self._tables.items()}

    def handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Args:
            event: The event object.
        """
        if results := event.results:
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
                self._callback(event)

    @property
    def event_types(self) -> set[EnOptEventType]:
        """Return the event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EnOptEventType.FINISHED_EVALUATION}

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
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
        return self._tables[key].get_table(self._sep)

    def add_column(self, table: str, name: str, title: str) -> None:
        """Add a column to a given table.

        Args:
            table: The name of the table to add the column to.
            name:  The name of the field to add as a column, using attribute syntax.
            title: The title of the column to add.
        """
        self._tables[table].add_column(name, title)


class _ResultsTable:
    def __init__(
        self,
        columns: dict[str, str],
        table_type: Literal["functions", "gradients"],
        domain: DomainType = "user",
    ) -> None:
        self._columns = columns
        self._results_type = table_type
        self._domain = domain
        self._frames: list[pd.DataFrame] = []

    @property
    def domain(self) -> DomainType:
        return self._domain

    def add_column(self, name: str, title: str) -> None:
        self._columns[name] = title

    def add_results(self, results: Sequence[Results]) -> bool:
        frame = results_to_dataframe(
            results, set(self._columns), result_type=self._results_type
        )
        if not frame.empty:
            self._frames.append(frame)
            return True
        return False

    def get_table(self, sep: str) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame()
        data = pd.concat(self._frames)
        reordered_columns = [
            name
            for key in self._columns
            for name in data.columns.to_numpy()
            if name == key or (isinstance(name, tuple) and name[0] == key)
        ]
        data = data.reindex(columns=reordered_columns)
        renamed_columns = [
            (str(self._columns[name[0]]), *(str(item) for item in name[1:]))
            if isinstance(name, tuple)
            else self._columns[name]
            for name in data.columns.to_numpy()
        ]
        data.columns = [
            sep.join(item) if isinstance(item, tuple) else item
            for item in renamed_columns
        ]
        return data.reset_index()
