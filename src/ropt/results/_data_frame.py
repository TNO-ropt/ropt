"""Convert Results objects to pandas DataFrames."""

from __future__ import annotations

from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING, Final, Literal

from ropt.exceptions import UnsupportedError

from ._frame_core import FRAME_SPECS, _get_select, _get_value, _has_results
from ._function_results import FunctionResults
from ._gradient_results import GradientResults

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.results import Results

_HAVE_PANDAS: Final = find_spec("pandas") is not None

if _HAVE_PANDAS:
    import pandas as pd


def _get_results(
    results: Results,
    sub_fields: set[str],
    result_type: Literal["functions", "gradients"],
) -> pd.DataFrame:
    if not sub_fields or not _has_results(results, result_type):
        return pd.DataFrame()

    return _join_frames(
        *(
            results.to_dataframe(
                spec.field,
                select=_get_select(spec.field, sub_fields),
                unstack=spec.unstack,
            ).rename(columns=partial(_add_prefix, prefix=spec.field))
            for spec in FRAME_SPECS[result_type]
            if getattr(results, spec.field, None) is not None
        )
    )


def _join_frames(*args: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in args if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return (
        frames[0].join(list(frames[1:]), how="outer") if len(frames) > 1 else frames[0]
    )


def _add_prefix(name: tuple[str, ...] | str, prefix: str) -> tuple[str, ...] | str:
    return (
        (f"{prefix}.{name[0]}", *name[1:])
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


def results_to_dataframe(
    results: Sequence[Results],
    fields: set[str],
    result_type: Literal["functions", "gradients"],
) -> pd.DataFrame:
    """Aggregate multiple results into a single pandas DataFrame.

    Concatenates the specified fields from a sequence of
    [`FunctionResults`][ropt.results.FunctionResults] or
    [`GradientResults`][ropt.results.GradientResults] objects. Fields are
    selected using dot notation (e.g., `evaluations.variables`); nested
    `metadata` entries are accessed as `evaluations.metadata.key`.
    Multi-dimensional fields are automatically unstacked into tuple-named
    columns.

    See [Working with Results](../optimizer_setup/results.md#exporting-to-pandas) for
    further details and examples.

    Args:
        results:     A sequence of [`Results`][ropt.results.Results] objects.
        fields:      Field names to include (dot notation for nested fields).
        result_type: `"functions"` or `"gradients"`.

    Returns:
        A DataFrame with one row per result and requested fields as columns.

    Raises:
        TypeError:        If `result_type` is invalid or results contain
                          unexpected types.
        UnsupportedError: If the `pandas` module is not installed.
    """
    if not _HAVE_PANDAS:
        msg = "results_to_dataframe requires the pandas module; install ropt[pandas]."
        raise UnsupportedError(msg)

    if result_type not in {"functions", "gradients"}:
        msg = f"Invalid frame output type: {result_type}"
        raise TypeError(msg)

    frames: list[pd.DataFrame] = []
    for item in results:
        if not isinstance(item, (FunctionResults, GradientResults)):
            msg = f"Invalid result type: {type(item)}"
            raise TypeError(msg)

        if _has_results(item, result_type):
            frames.append(
                _add_metadata(_get_results(item, fields, result_type), item, fields)
            )

    return pd.concat(frames) if frames else pd.DataFrame()
