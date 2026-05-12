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
    """This event handler tracks results and stores them in pandas DataFrames.

    **Tables**

    Tables can be generated for
    [`FunctionsResults`][ropt.results.FunctionResults] and
    [`GradientsResults`][ropt.results.GradientResults] respectively. Tables are
    added via the `add_table` method, which takes a name, a type (either
    "functions" or "gradients"), a column specification and an optional domain
    type. The column specification determines which fields of the results are
    stored in the table and how they are named. The domain type determines
    whether the results are transformed to the user domain before being stored
    in the table.

    Tables are accessed by their name in attributes, for example, as
    `handler["evaluations"]`.

    Warning:
        Tables are generated on the fly from internal data when accessing them
        in this way. When multiple access are needed, it is more efficient to
        first store them in a variable.

    **Column specification**

    Columns are specified by providing a dictionary that maps field names to
    column titles. The keys denote the names of the fields, using attribute
    syntax. For instance a `function.objectives` key indicates the the
    result should contain a column with objective values that are found in the
    `objectives` field of the `function` field of the result. The values
    corresponding to the keys are used to provide the column names.

    For example, passing this dictionary via the `columns` argument generates a
    table containing the batch id, the values of all calculated objectives and
    the vector of variables.

    ```python
    {
        "batch_id": "Batch",
        "functions.objectives": "Objective",
        "evaluations.variables": "Variables",
    }
    ```

    Some fields may result in multiple columns in the DataFrame if their values
    are vectors or matrices. For example, `evaluations.variables` will generate
    a separate column for each variable. The table specification above may
    generate a pandas dataframe looking something like this:

    ```
        Batch   Objective,0  Variables,v0  Variables,v1  Variables,v2
    0       0  1.309826e+02      0.500000      0.900000      1.300000
    1       0  4.362553e+12    120.900265     20.698539    -90.578972
    ...
    ```

    Here, because the variables are vectors of length 2, there are two variable
    columns generated. The corresponding column names consist of the column
    title and the name of the variable vector, separated by a comma. Note that
    the `function.objectives` column also contains a comma followed by a 0
    value. This is because the `functions.objectives` is also a vector of
    values, there just happens to be only one objective. Its index is used
    instead of a name, because no name was provided in the configuration of the
    optimization. Fields may even have matrix values, in which case the column
    names may be contain two item names or indices separated by commas.

    Tip: Changing the column name separator.
        By default a comma is used to separate fields in the column names if
        needed. The `sep` input can be used to provide an alternative separator.

        You can exploit this by specifying a newline as the separator and
        display a nicely formatted table using the `tabulate` package:

        ```python
        from tabulate import tabulate

        print(tabulate(table["functions"], headers="keys", showindex=False))
        ```

        which will show something like this using multi-line headers:
        ```
          Batch         Objective    Variables    Variables     Variables
                                0           v0           v1            v2
        -------  ----------------  -----------  -----------  ------------
              0           130.983          0.5          0.9           1.3
        ...
        ```


    **Default tables**

    The `set_default_tables` method can be used to add a set of default tables:

    - For functions results it generates these tables:
        - `"functions"`: contains a set of values of the calculated functions.
        - `"evaluations"`: contains a set of values for all evaluations.
        - `"constraints"`: contains a set of values for all constraints.
    - For gradients results it generates these tables:
        - `"gradients"`: contains a set of values of the calculated gradients.
        - `"perturbations"`: contains a set of values for all perturbations.


    **Callback functionality**

    The tables are updated anytime a result is processed. To be able to do
    something with the tables each time they are updated a callback can be
    provided set using `set_callback`. This callback will called anytime the
    tables are updated, passing the event that caused the tables to be updated.
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
