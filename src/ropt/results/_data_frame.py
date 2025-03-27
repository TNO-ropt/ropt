"""Generate a results report in a `pandas` data frame."""

from __future__ import annotations

from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final, Literal, Sequence

from ropt.enums import ResultAxis

from ._function_results import FunctionResults
from ._gradient_results import GradientResults

if TYPE_CHECKING:
    from ropt.results import Results

_HAVE_PANDAS: Final = find_spec("pandas") is not None

if TYPE_CHECKING:
    from ropt.results import Results

if _HAVE_PANDAS:
    import pandas as pd


def _get_function_results(
    results: Results,
    sub_fields: set[str],
    names: dict[str, Sequence[str | int] | None] | None,
) -> pd.DataFrame:
    if (
        not sub_fields
        or not isinstance(results, FunctionResults)
        or results.functions is None
    ):
        return pd.DataFrame()

    functions = results.to_dataframe(
        "functions",
        select=_get_select("functions", sub_fields),
        unstack=[ResultAxis.OBJECTIVE, ResultAxis.NONLINEAR_CONSTRAINT],
        names=names,
    ).rename(columns=partial(_add_prefix, prefix="functions"))

    evaluations = results.to_dataframe(
        "evaluations",
        select=_get_select("evaluations", sub_fields),
        unstack=[
            ResultAxis.VARIABLE,
            ResultAxis.OBJECTIVE,
            ResultAxis.NONLINEAR_CONSTRAINT,
        ],
        names=names,
    ).rename(columns=partial(_add_prefix, prefix="evaluations"))

    if results.constraint_info is not None:
        constraint_info = results.to_dataframe(
            "constraint_info",
            select=_get_select("constraint_info", sub_fields),
            unstack=[
                ResultAxis.VARIABLE,
                ResultAxis.LINEAR_CONSTRAINT,
                ResultAxis.NONLINEAR_CONSTRAINT,
            ],
            names=names,
        ).rename(columns=partial(_add_prefix, prefix="constraint_info"))

        return _join_frames(functions, evaluations, constraint_info)

    return _join_frames(functions, evaluations)


def _get_gradient_results(
    results: Results,
    sub_fields: set[str],
    names: dict[str, Sequence[str | int] | None] | None,
) -> pd.DataFrame:
    if (
        not sub_fields
        or not isinstance(results, GradientResults)
        or results.gradients is None
    ):
        return pd.DataFrame()

    gradients = results.to_dataframe(
        "gradients",
        select=_get_select("gradients", sub_fields),
        unstack=[
            ResultAxis.OBJECTIVE,
            ResultAxis.NONLINEAR_CONSTRAINT,
            ResultAxis.VARIABLE,
        ],
        names=names,
    ).rename(columns=partial(_add_prefix, prefix="gradients"))

    evaluations = results.to_dataframe(
        "evaluations",
        select=_get_select("evaluations", sub_fields),
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


def _get_select(field_name: str, sub_fields: set[str]) -> list[str]:
    return [
        item.removeprefix(f"{field_name}.")
        for item in sub_fields
        if item.startswith(f"{field_name}.")
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


def results_to_dataframe(
    results: Sequence[Results],
    fields: set[str],
    result_type: Literal["functions", "gradients"],
    names: dict[str, Sequence[str | int] | None] | None = None,
) -> pd.DataFrame:
    """Combine a sequence of results into a single pandas DataFrame.

    This function aggregates results from multiple
    [`FunctionResults`][ropt.results.FunctionResults] or
    [`GradientResults`][ropt.results.GradientResults] objects into a single
    `pandas` DataFrame. It is designed to be used with observers that produce
    results during the optimization process.

    The `fields` argument determines which data fields to include in the
    DataFrame. These fields can be any of the attributes defined within
    [`FunctionResults`][ropt.results.FunctionResults] or
    [`GradientResults`][ropt.results.GradientResults]. Nested fields are
    specified using dot notation (e.g., `evaluations.variables` to include the
    `variables` field within the `evaluations` object).

    The `evaluation_info` sub-fields, found within the `evaluations` fields of
    [`functions`][ropt.results.FunctionEvaluations] and
    [`gradient`][ropt.results.GradientEvaluations] results, respectively, are
    dictionaries. To include specific keys from these dictionaries, use the
    format `evaluations.evaluation_info.key`, where `key` is the name of the
    desired key.

    Many fields may result in multiple columns in the DataFrame. For example,
    `evaluations.variables` will generate a separate column for each variable.
    If available, variable names will be used as column labels.
    Multi-dimensional fields, such as those with named realizations and
    objectives, will have column names that are tuples of the corresponding
    names.

    The `result_type` argument specifies whether to include function evaluation
    results (`functions`) or gradient results (`gradients`).

    The `names` argument is an optional dictionary that maps axis types to
    names. These names are used to label the multi-index columns in the
    resulting DataFrame. If not provided, numerical indices are used.

    Args:
        results:     A sequence of [`Results`][ropt.results.Results] objects
                     to combine.
        fields:      The names of the fields to include in the DataFrame.
        result_type: The type of results to include ("functions" or
                     "gradients").
        names:       A dictionary mapping axis types to names.

    Returns:
        A `pandas` DataFrame containing the combined results.
    """
    if result_type not in ("functions", "gradients"):
        msg = f"Invalid frame output type: {result_type}"
        raise TypeError(msg)

    frame = pd.DataFrame()
    for item in results:
        if not isinstance(item, (FunctionResults, GradientResults)):
            msg = f"Invalid result type: {type(item)}"
            raise TypeError(msg)

        if (
            result_type == "functions"
            and isinstance(item, FunctionResults)
            and item.functions is not None
        ):
            frame = pd.concat(
                [
                    frame,
                    _add_metadata(
                        _get_function_results(item, fields, names), item, fields
                    ),
                ]
            )
        elif (
            result_type == "gradients"
            and isinstance(item, GradientResults)
            and item.gradients is not None
        ):
            frame = pd.concat(
                [
                    frame,
                    _add_metadata(
                        _get_gradient_results(item, fields, names), item, fields
                    ),
                ]
            )

    return frame
