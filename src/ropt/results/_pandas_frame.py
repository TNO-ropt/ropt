"""Convert Results objects to pandas DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ropt.exceptions import UnsupportedError

from ._frame_core import UNSTACK_AXES, _has_results, _value_fields
from ._frame_support import HAVE_PANDAS, missing_engine_message
from ._function_results import FunctionResults
from ._gradient_results import GradientResults

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.results import Results

if HAVE_PANDAS:
    import pandas as pd

    from ._pandas import _to_pandas_frame


def _get_results(
    results: Results,
    sub_fields: set[str],
    result_type: Literal["functions", "gradients"],
) -> pd.DataFrame:
    if not sub_fields or not _has_results(results, result_type):
        return pd.DataFrame()
    return _to_pandas_frame(
        results, _value_fields(sub_fields), UNSTACK_AXES[result_type], aggregated=True
    )


def results_to_pandas(
    results: Sequence[Results],
    fields: set[str],
    result_type: Literal["functions", "gradients"],
) -> pd.DataFrame:
    """Aggregate multiple results into a single pandas DataFrame.

    Concatenates the specified fields from a sequence of
    [`FunctionResults`][ropt.results.FunctionResults] or
    [`GradientResults`][ropt.results.GradientResults] objects, one row per
    result. See [Aggregating multiple results](../optimizer_setup/results.md#aggregating-multiple-results)
    for field selection and unstacking.

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
    if not HAVE_PANDAS:
        msg = missing_engine_message(
            "pandas", "results_to_pandas", "use results_to_polars"
        )
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
            frames.append(_get_results(item, fields, result_type))

    return pd.concat(frames) if frames else pd.DataFrame()
