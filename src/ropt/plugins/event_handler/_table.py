from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import pandas as pd

from ropt.enums import EventType
from ropt.plugins.event_handler.base import EventHandler
from ropt.results import Results, results_to_dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.workflow import Event

_FUNCTION_TABLES: Final[dict[str, dict[str, str]]] = {
    "functions": {
        "batch_id": "Batch",
        "functions.weighted_objective": "Total-Objective",
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
        "gradients.weighted_objective": "Total-Gradient",
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


class DefaultTableHandler(EventHandler):
    """Default handler for generating result tables."""

    def __init__(
        self,
        *,
        functions: dict[str, dict[str, str]] | None = None,
        gradients: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Initialize a default table event handler.

        Tables are accessed by their name in attributes, i.e. if the `functions`
        input has an entry `evaluations`, this table can be accessed via
        `handler["evaluations"]`. Note that the tables are generated on the fly
        from internal data when accessing them in this way. When multiple access
        are needed, it is more efficient to first store them in a variable.

        The column names are formed from the field names of the results. Many
        fields may result in multiple columns in the DataFrame. For example,
        `evaluations.variables` will generate a separate column for each
        variable. If available, variable names will be used as column labels.
        Multi-dimensional fields, such as those with named realizations and
        objectives, will have column names that are tuples of the corresponding
        names.

        The stored tables contain human-readable column names in their `attrs`
        fields that are derived from the titles given in the input dictionaries:

        - "renamed_columns":   The column names are renamed according to the
                               inputs dictionaries, and compound names are still
                               tuples.
        - "formatted_columns": The column names are renamed, and compound names
                               are joined by newlines.

        These column names can be assigned to the column fields of the tables.

        Tip: Multi-line column names.
            The column names obtained from the attributes via the "formatted_columns"
            key may contain new lines. These can be displayed well using the
            `tabulate` package:

            ```python
            from tabulate import tabulate

            table = table["functions"]
            table.columns = table.attrs["formatted_columns"]
            print(tabulate(table, headers="keys", tablefmt="simple", showindex=False))
            ```

        Args:
            functions:   Dictionary of tables with function results.
            gradients:   Dictionary of tables with gradient results.
        """
        super().__init__()
        if functions is None:
            functions = _FUNCTION_TABLES
        if gradients is None:
            gradients = _GRADIENT_TABLES
        self._tables = {
            table_name: _ResultsTable(functions[table_name], table_type="functions")
            for table_name in functions
        }
        self._tables.update(
            {
                table_name: _ResultsTable(gradients[table_name], table_type="gradients")
                for table_name in gradients
            }
        )

    def handle_event(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: The event object.
        """
        if (results := event.data.get("results")) is None:
            return
        transforms = event.data["transforms"]
        results = tuple(
            item
            if transforms is None
            else item.transform_from_optimizer(event.data["config"], transforms)
            for item in results
        )
        for table in self._tables.values():
            table.add_results(results)

    @property
    def event_types(self) -> set[EventType]:
        """Return the event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EventType.FINISHED_EVALUATION}

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Retrieve a of a table from the event handler.

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


class _ResultsTable:
    def __init__(
        self,
        columns: dict[str, str],
        table_type: Literal["functions", "gradients"],
    ) -> None:
        self._columns = columns
        self._results_type = table_type
        self._frames: list[pd.DataFrame] = []
        self._attrs: dict[str, Any] = {}

    def add_results(self, results: Sequence[Results]) -> None:
        frame = results_to_dataframe(
            results, set(self._columns), result_type=self._results_type
        )
        if not frame.empty:
            self._frames.append(frame)

    def get_table(self) -> pd.DataFrame | None:
        data = pd.concat(self._frames)
        data = self._reorder_columns(data.reset_index())
        renamed_columns = self._get_columns(data)
        data.attrs = {
            "renamed_columns": renamed_columns,
            "formatted_columns": [
                "\n".join(item) if isinstance(item, tuple) else item
                for item in renamed_columns
            ],
        }
        return data

    def _reorder_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        reordered_columns = [
            name
            for key in self._columns
            for name in data.columns.to_numpy()
            if name == key or (isinstance(name, tuple) and name[0] == key)
        ]
        return data.reindex(columns=reordered_columns)

    def _get_columns(self, data: pd.DataFrame) -> list[str | tuple[str, ...]]:
        return [
            (str(self._columns[name[0]]), *(str(item) for item in name[1:]))
            if isinstance(name, tuple)
            else self._columns[name]
            for name in data.columns.to_numpy()
        ]
