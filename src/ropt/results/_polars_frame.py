"""Convert Results objects to polars DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ropt.exceptions import UnsupportedError

from ._frame_core import FRAME_SPECS, _get_select, _get_value, _has_results
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
    sub_fields: set[str],
    result_type: Literal["functions", "gradients"],
    sep: str,
) -> pl.DataFrame:
    if not sub_fields or not _has_results(results, result_type):
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    keys: list[str] = []
    for spec in FRAME_SPECS[result_type]:
        if getattr(results, spec.field, None) is None:
            continue
        frame, key_columns = _to_polars_frame(
            results,
            spec.field,
            _get_select(spec.field, sub_fields),
            spec.unstack,
            sep,
        )
        frames.append(_add_prefix(frame, key_columns, spec.field))
        keys += [column for column in key_columns if column not in keys]
    return _join_frames(frames, keys)


def _join_frames(args: Sequence[pl.DataFrame], keys: Sequence[str]) -> pl.DataFrame:
    frames = [frame for frame in args if frame.height > 0]
    if not frames:
        return pl.DataFrame()
    joined_frame = frames[0]
    for frame in frames[1:]:
        joined_frame = joined_frame.join(
            frame,
            on=[
                column
                for column in frame.columns
                if column in keys and column in joined_frame.columns
            ],
            how="full",
            coalesce=True,
            maintain_order="left_right",
        )
    return joined_frame.select(
        *(column for column in joined_frame.columns if column in keys),
        *(column for column in joined_frame.columns if column not in keys),
    )


def _add_prefix(frame: pl.DataFrame, keys: Sequence[str], prefix: str) -> pl.DataFrame:
    return frame.rename(
        {column: f"{prefix}.{column}" for column in frame.columns if column not in keys}
    )


def _add_metadata(
    frame: pl.DataFrame, results: Results, sub_fields: set[str]
) -> pl.DataFrame:
    if frame.height == 0:
        return frame
    for field in sub_fields:
        split_fields = field.split(".")
        if split_fields[0] == "metadata":
            value = _get_value(results.metadata, split_fields[1:])
            if value is not None:
                frame = frame.with_columns(pl.lit(value).alias(field))
    return frame


def results_to_polars(
    results: Sequence[Results],
    fields: set[str],
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
        fields:      Field names to include (dot notation for nested fields).
        result_type: `"functions"` or `"gradients"`.
        sep:         Separator used to join unstacked column names.

    Returns:
        A DataFrame with one row per result and requested fields as columns.

    Raises:
        TypeError:        If `result_type` is invalid or results contain
                          unexpected types.
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

    frames: list[pl.DataFrame] = []
    for item in results:
        if not isinstance(item, (FunctionResults, GradientResults)):
            msg = f"Invalid result type: {type(item)}"
            raise TypeError(msg)

        if _has_results(item, result_type):
            frames.append(
                _add_metadata(
                    _get_results(item, fields, result_type, sep), item, fields
                )
            )

    frames = [frame for frame in frames if frame.width > 0]
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
