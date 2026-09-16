"""Convert Results objects to polars DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ropt.exceptions import UnsupportedError

from ._frame_core import (
    _UNORDERED_FIELDS_ERROR,
    UNSTACK_AXES,
    _duplicate_fields,
    _has_results,
    _is_unordered,
    _value_fields,
)
from ._frame_support import HAVE_POLARS, missing_engine_message
from ._function_results import FunctionResults
from ._gradient_results import GradientResults

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.results import Results

if HAVE_POLARS:
    import polars as pl

    from ._polars import _to_polars_frame


def _get_results(
    results: Results,
    sub_fields: Sequence[str],
    result_type: Literal["functions", "gradients"],
    sep: str,
) -> pl.DataFrame:
    if not sub_fields or not _has_results(results, result_type):
        return pl.DataFrame()
    return _to_polars_frame(
        results,
        _value_fields(sub_fields),
        UNSTACK_AXES[result_type],
        sep,
        aggregated=True,
    )[0]


def results_to_polars(
    results: Sequence[Results],
    fields: Sequence[str],
    result_type: Literal["functions", "gradients"],
    sep: str = ",",
) -> pl.DataFrame:
    """Aggregate multiple results into a single polars DataFrame.

    This is the polars counterpart of
    [`results_to_pandas`][ropt.results.results_to_pandas], returned in
    long format with tuple column names joined into a single string using
    `sep`. Unlike the pandas export, fields with different granularities are
    aligned into one table rather than kept as separate blocks. See
    [Exporting to polars](../optimizer_setup/results.md#exporting-to-polars)
    for details.

    Args:
        results:     A sequence of [`Results`][ropt.results.Results] objects.
        fields:      Field names to include, in column order (dot notation for
                     nested fields).
        result_type: `"functions"` or `"gradients"`.
        sep:         Separator used to join unstacked column names.

    Returns:
        A DataFrame with the requested fields as columns, keyed by `batch_id`
        and by any axes that stay stacked.

    Raises:
        TypeError:        If `result_type` is invalid, if `fields` is a set
                          rather than an ordered sequence, or if results
                          contain unexpected types.
        ValueError:       If `fields` names the same path more than once.
        UnsupportedError: If the `polars` module is not installed.
    """
    if not HAVE_POLARS:
        msg = missing_engine_message(
            "polars", "results_to_polars", "use results_to_pandas"
        )
        raise UnsupportedError(msg)

    if result_type not in {"functions", "gradients"}:
        msg = f"Invalid frame output type: {result_type}"
        raise TypeError(msg)

    if _is_unordered(fields):
        raise TypeError(_UNORDERED_FIELDS_ERROR)
    duplicates = _duplicate_fields(fields)
    if duplicates:
        msg = f"Duplicate fields: {duplicates}"
        raise ValueError(msg)
    frames: list[pl.DataFrame] = []
    for item in results:
        if not isinstance(item, (FunctionResults, GradientResults)):
            msg = f"Invalid result type: {type(item)}"
            raise TypeError(msg)

        if _has_results(item, result_type):
            frames.append(_get_results(item, fields, result_type, sep))

    frames = [frame for frame in frames if frame.width > 0]
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
