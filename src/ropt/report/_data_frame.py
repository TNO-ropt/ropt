"""Generate a results report in a `pandas` data frame."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Iterable, Literal

from ropt.enums import ResultAxisName
from ropt.results import FunctionResults, GradientResults

from ._utils import _HAVE_PANDAS, _add_metadata, _add_prefix, _get_select

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig
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
    fields of the [`Results`][ropt.results.Results] objects passed during each call
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
        If available, names specified in the optimizer configuration, such as
        variable names, will be used as column labels. Because the exported fields
        may be multi-dimensional with names defined along each axis, for instance,
        realizations and objectives, which both can be named, the final name may
        consist of a tuple of names.

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

    def add_results(self, config: EnOptConfig, results: Iterable[Results]) -> bool:
        """Add results to the table.

        This method can be called directly from any observers connected to
        events that produce results.

        Args:
            config:  The configuration of the optimizer generating the results.
            results: The results to add.

        Returns:
            True if a result was added, else False
        """
        added = False
        for item in results:
            if (
                self._table_type == "functions"
                and isinstance(item, FunctionResults)
                and item.functions is not None
            ):
                frame = _get_function_results(config, item, self._fields)
            elif (
                self._table_type == "gradients"
                and isinstance(item, GradientResults)
                and item.gradients is not None
            ):
                frame = _add_gradient_results(config, item, self._fields)
            else:
                continue
            if not frame.empty:
                frame = _add_metadata(frame, item, self._fields)
                self._frame = pd.concat([self._frame, frame])
                added = True
        return added

    @property
    def frame(self) -> pd.DataFrame:
        """Return the function results generated so far.

        Returns:
            A pandas data frame with the results.
        """
        return self._frame


def _get_function_results(
    config: EnOptConfig, results: Results, sub_fields: set[str]
) -> pd.DataFrame:
    if (
        not sub_fields
        or not isinstance(results, FunctionResults)
        or results.functions is None
    ):
        return pd.DataFrame()

    functions = results.to_dataframe(
        config,
        "functions",
        select=_get_select(results, "functions", sub_fields),
        unstack=[ResultAxisName.OBJECTIVE, ResultAxisName.NONLINEAR_CONSTRAINT],
    ).rename(columns=partial(_add_prefix, prefix="functions"))

    bound_constraints = (
        pd.DataFrame()
        if results.bound_constraints is None
        else results.to_dataframe(
            config,
            "bound_constraints",
            select=_get_select(results, "bound_constraints", sub_fields),
            unstack=[ResultAxisName.VARIABLE],
        ).rename(columns=partial(_add_prefix, prefix="bound_constraints"))
    )

    linear_constraints = (
        pd.DataFrame()
        if results.linear_constraints is None
        else results.to_dataframe(
            config,
            "linear_constraints",
            select=_get_select(results, "linear_constraints", sub_fields),
            unstack=[ResultAxisName.LINEAR_CONSTRAINT],
        ).rename(columns=partial(_add_prefix, prefix="linear_constraints"))
    )

    nonlinear_constraints = (
        pd.DataFrame()
        if results.nonlinear_constraints is None
        else results.to_dataframe(
            config,
            "nonlinear_constraints",
            select=_get_select(results, "nonlinear_constraints", sub_fields),
            unstack=[ResultAxisName.NONLINEAR_CONSTRAINT],
        ).rename(columns=partial(_add_prefix, prefix="nonlinear_constraints"))
    )

    evaluations = results.to_dataframe(
        config,
        "evaluations",
        select=_get_select(results, "evaluations", sub_fields),
        unstack=[
            ResultAxisName.VARIABLE,
            ResultAxisName.OBJECTIVE,
            ResultAxisName.NONLINEAR_CONSTRAINT,
        ],
    ).rename(columns=partial(_add_prefix, prefix="evaluations"))

    return _join_frames(
        functions,
        bound_constraints,
        linear_constraints,
        nonlinear_constraints,
        evaluations,
    )


def _add_gradient_results(
    config: EnOptConfig, results: Results, sub_fields: set[str]
) -> pd.DataFrame:
    if (
        not sub_fields
        or not isinstance(results, GradientResults)
        or results.gradients is None
    ):
        return pd.DataFrame()

    gradients = results.to_dataframe(
        config,
        "gradients",
        select=_get_select(results, "gradients", sub_fields),
        unstack=[
            ResultAxisName.OBJECTIVE,
            ResultAxisName.NONLINEAR_CONSTRAINT,
            ResultAxisName.VARIABLE,
        ],
    ).rename(columns=partial(_add_prefix, prefix="gradients"))

    evaluations = results.to_dataframe(
        config,
        "evaluations",
        select=_get_select(results, "evaluations", sub_fields),
        unstack=[
            ResultAxisName.VARIABLE,
            ResultAxisName.OBJECTIVE,
            ResultAxisName.NONLINEAR_CONSTRAINT,
        ],
    ).rename(columns=partial(_add_prefix, prefix="evaluations"))

    return _join_frames(gradients, evaluations)


def _join_frames(*args: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in args if not frame.empty]
    return (
        frames[0].join(list(frames[1:]), how="outer") if len(frames) > 1 else frames[0]
    )
