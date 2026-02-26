from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import pandas as pd

from ropt.enums import EventType
from ropt.plugins.event_handler.base import EventHandler
from ropt.results import Results, results_to_dataframe

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
    """This event handler tracks results and stores them in pandas DataFrames.

    **Tables**

    The `functions` and `gradients` inputs are dictionaries where each item
    denotes a table to generate, for
    [`FunctionsResults`][ropt.results.FunctionResults] and
    [`GradientsResults`][ropt.results.GradientResults] respectively. Tables are
    accessed by their name in attributes, i.e. if the `functions` input has an
    entry `evaluations`, this table can be accessed via
    `handler["evaluations"]`. Note that the tables are generated on the fly from
    internal data when accessing them in this way. When multiple access are
    needed, it is more efficient to first store them in a variable.

    By default the `functions` and `gradients` are set to `None` directing the handler
    to generate a set of default tables:

    - For the `functions` input:
        - `"functions"`: contains a set of values of the calculated functions.
        - `"evaluations"`: contains a set of values for all evaluations.
        - `"constraints"`: contains a set of values for all constraints.
    - For the `gradients` input:
        - `"gradients"`: contains a set of values of the calculated gradients.
        - `"perturbations"`: contains a set of values for all perturbations.

    To change (or disable) the generation of these table, pass an appropriate
    (or empty) dictionary specifying the desired tables.

    **Table specification**

    Each value of each entry in the `functions` and `gradients` dictionaries is
    another dictionary that determines which fields in the results should be
    stored in the corresponding table. The keys denote the names of the fields,
    using attribute syntax. For instance a `result.function.objectives` key
    indicates the the result should contain a column with objective values that
    are found in the `objectives` field of the `function` field of the result.
    The values corresponding to the keys are used to provide the column names.

    For example, passing this dictionary via the `functions` input generates a
    table containing the batch id, the values of all calculated objectives and
    the vector of variables.

    ```python
    {
        "functions": {
            "batch_id": "Batch",
            "functions.objectives": "Objective",
            "evaluations.variables": "Variables",
        }
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

    **Callback functionality**

    The tables are updated anytime a result is processed. To be able to do
    something with the tables each time they are updated a callback can be
    provided via the `callback` input. This callback will called anytime the
    tables are update, passing the handler as its only argument.
    """

    def __init__(
        self,
        *,
        functions: dict[str, dict[str, str]] | None = None,
        gradients: dict[str, dict[str, str]] | None = None,
        sep: str = ",",
        callback: Callable[[EventHandler], None] | None = None,
    ) -> None:
        """Initialize a default table event handler.

        Args:
            functions: Dictionary of tables with function results.
            gradients: Dictionary of tables with gradient results.
            sep:       Separator used in column names.
            callback:  An optional callback that is called when the tables are update.
        """
        super().__init__()
        self._sep = sep
        self._callback = callback
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
        if (
            any(table.add_results(results) for table in self._tables.values())
            and self._callback is not None
        ):
            self._callback(self)

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
        return self._tables[key].get_table(self._sep)


class _ResultsTable:
    def __init__(
        self,
        columns: dict[str, str],
        table_type: Literal["functions", "gradients"],
    ) -> None:
        self._columns = columns
        self._results_type = table_type
        self._frames: list[pd.DataFrame] = []

    def add_results(self, results: Sequence[Results]) -> bool:
        frame = results_to_dataframe(
            results, set(self._columns), result_type=self._results_type
        )
        if not frame.empty:
            self._frames.append(frame)
            return True
        return False

    def get_table(self, sep: str) -> pd.DataFrame | None:
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
        return data
