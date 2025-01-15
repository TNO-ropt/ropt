"""Generate a results report in a `pandas` data frame."""

from __future__ import annotations

from dataclasses import fields
from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final, Literal, Sequence

from ropt.enums import ResultAxis
from ropt.results import FunctionResults, GradientResults

if TYPE_CHECKING:
    from ropt.results import Results

_HAVE_PANDAS: Final = find_spec("pandas") is not None

if TYPE_CHECKING:
    from ropt.results import Results

if _HAVE_PANDAS:
    import pandas as pd


class ResultsDataFrame:
    """Generate a results report in a `pandas` DataFrame.

    The class is intended to be used for gathering results from
    [`Results`][ropt.results.Results] objects and storing them as a
    [`pandas`](https://pandas.pydata.org/) DataFrame.

    New results can be added to the stored DataFrame as they become available
    using the `add_results` method. The content of the DataFrame is taken from the
    fields of the [`Results`][ropt.results.Results] object passed during each call
    to `add_results`. The updated table can be retrieved at any time via the
    [`frame`][ropt.report.ResultsDataFrame.frame] property.
    """

    def __init__(
        self,
        fields: set[str],
        *,
        table_type: Literal["functions", "gradients"] = "functions",
    ) -> None:
        """Initialize a ResultsDataFrame object.

        The set of fields used is determined by the `results_fields` argument
        passed when initializing the object. These can be any of the fields
        defined in a [`FunctionResults`][ropt.results.FunctionResults] or a
        [`GradientResults`][ropt.results.GradientResults]. Most of the fields in
        these objects are nested, and the fields to export must be specified
        specifically in the form `field.subfield`. For instance, to specify the
        `variables` field of the `evaluations` field from a function result,
        the specification would be `evaluations.variables`.

        Note that many fields may, in fact, generate multiple columns in the
        resulting data frame. For instance, when specifying
        `evaluations.variables`, a column will be generated for each variable.
        If available, variable names, will be used as column labels. Because
        the exported fields may be multi-dimensional with names defined along
        each axis, for instance, realizations and objectives, which both
        can be named, the final name may consist of a tuple of names.

        The `table_type` argument is used to determine which type of results
        should be reported: either function evaluation results (`functions`) or
        gradient results (`gradients`).

        Args:
            fields:     The fields of the results to store.
            table_type: The type of the table.
        """
        if not _HAVE_PANDAS:
            msg = "ResultsDataFrame requires the `pandas` modules"
            raise NotImplementedError(msg)

        self._fields = fields
        self._table_type = table_type
        self._frame = pd.DataFrame()

    def add_results(
        self,
        results: Results,
        names: dict[str, Sequence[str] | None] | None = None,
    ) -> bool:
        """Add a results object to the table.

        This method can be called directly from any observers connected to
        events that produce results.

        The `names` argument is an optional dictionary that maps axis types to
        names, that are used to label the multi-index columns in the resulting
        data frame. If not provided, numerical indices are used.

        Args:
            results: The results to add.
            names:   A dictionary mapping axis types to names.

        Returns:
            True if a result was added, else False
        """
        if not isinstance(results, (FunctionResults, GradientResults)):
            msg = "ResultsDataFrame.add_results() requires FunctionResults or GradientResults"
            raise TypeError(msg)

        frame: pd.DataFrame | None = None
        if (
            self._table_type == "functions"
            and isinstance(results, FunctionResults)
            and results.functions is not None
        ):
            frame = _get_function_results(results, self._fields, names)
        elif (
            self._table_type == "gradients"
            and isinstance(results, GradientResults)
            and results.gradients is not None
        ):
            frame = _get_gradient_results(results, self._fields, names)
        if frame is not None:
            frame = _add_metadata(frame, results, self._fields)
            self._frame = pd.concat([self._frame, frame])
            return True

        return False

    @property
    def frame(self) -> pd.DataFrame:
        """Return the function results generated so far.

        Returns:
            A pandas data frame with the results.
        """
        return self._frame


def _get_function_results(
    results: Results,
    sub_fields: set[str],
    names: dict[str, Sequence[str] | None] | None,
) -> pd.DataFrame:
    if (
        not sub_fields
        or not isinstance(results, FunctionResults)
        or results.functions is None
    ):
        return pd.DataFrame()

    functions = results.to_dataframe(
        "functions",
        select=_get_select(results, "functions", sub_fields),
        unstack=[ResultAxis.OBJECTIVE, ResultAxis.NONLINEAR_CONSTRAINT],
        names=names,
    ).rename(columns=partial(_add_prefix, prefix="functions"))

    bound_constraints = (
        pd.DataFrame()
        if results.bound_constraints is None
        else results.to_dataframe(
            "bound_constraints",
            select=_get_select(results, "bound_constraints", sub_fields),
            unstack=[ResultAxis.VARIABLE],
            names=names,
        ).rename(columns=partial(_add_prefix, prefix="bound_constraints"))
    )

    linear_constraints = (
        pd.DataFrame()
        if results.linear_constraints is None
        else results.to_dataframe(
            "linear_constraints",
            select=_get_select(results, "linear_constraints", sub_fields),
            unstack=[ResultAxis.LINEAR_CONSTRAINT],
            names=names,
        ).rename(columns=partial(_add_prefix, prefix="linear_constraints"))
    )

    nonlinear_constraints = (
        pd.DataFrame()
        if results.nonlinear_constraints is None
        else results.to_dataframe(
            "nonlinear_constraints",
            select=_get_select(results, "nonlinear_constraints", sub_fields),
            unstack=[ResultAxis.NONLINEAR_CONSTRAINT],
            names=names,
        ).rename(columns=partial(_add_prefix, prefix="nonlinear_constraints"))
    )

    evaluations = results.to_dataframe(
        "evaluations",
        select=_get_select(results, "evaluations", sub_fields),
        unstack=[
            ResultAxis.VARIABLE,
            ResultAxis.OBJECTIVE,
            ResultAxis.NONLINEAR_CONSTRAINT,
        ],
        names=names,
    ).rename(columns=partial(_add_prefix, prefix="evaluations"))

    return _join_frames(
        functions,
        bound_constraints,
        linear_constraints,
        nonlinear_constraints,
        evaluations,
    )


def _get_gradient_results(
    results: Results,
    sub_fields: set[str],
    names: dict[str, Sequence[str] | None] | None,
) -> pd.DataFrame:
    if (
        not sub_fields
        or not isinstance(results, GradientResults)
        or results.gradients is None
    ):
        return pd.DataFrame()

    gradients = results.to_dataframe(
        "gradients",
        select=_get_select(results, "gradients", sub_fields),
        unstack=[
            ResultAxis.OBJECTIVE,
            ResultAxis.NONLINEAR_CONSTRAINT,
            ResultAxis.VARIABLE,
        ],
        names=names,
    ).rename(columns=partial(_add_prefix, prefix="gradients"))

    evaluations = results.to_dataframe(
        "evaluations",
        select=_get_select(results, "evaluations", sub_fields),
        unstack=[
            ResultAxis.VARIABLE,
            ResultAxis.OBJECTIVE,
            ResultAxis.NONLINEAR_CONSTRAINT,
        ],
        names=names,
    ).rename(columns=partial(_add_prefix, prefix="evaluations"))

    return _join_frames(gradients, evaluations)


def _join_frames(*args: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in args if not frame.empty]
    return (
        frames[0].join(list(frames[1:]), how="outer") if len(frames) > 1 else frames[0]
    )


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
