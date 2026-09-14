from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from ropt.exceptions import UnsupportedError

from ._frame_support import (
    HAVE_PANDAS,
    HAVE_POLARS,
    missing_engine_message,
)
from ._result_field import AxisMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ropt.enums import AxisName


if TYPE_CHECKING and HAVE_PANDAS:
    import pandas as pd  # ruff: ignore[typing-only-third-party-import]
if TYPE_CHECKING and HAVE_POLARS:
    import polars as pl  # ruff: ignore[typing-only-third-party-import]
if HAVE_PANDAS:
    from ._pandas import _to_pandas_frame
if HAVE_POLARS:
    from ._polars import _to_polars_frame

TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class Results(AxisMetadata, ABC):
    """Abstract base class for optimization results.

    Subclassed by [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults].

    See [Working with Results](../optimizer_setup/results.md) for a narrative overview.

    Attributes:
        batch_id: Identifier for the evaluation batch.
        metadata: Dictionary of additional information (not used internally).
        names:    Mapping from [`AxisName`][ropt.enums.AxisName] to label tuples
                  for DataFrame export.
    """

    batch_id: int
    metadata: dict[str, Any]
    names: dict[str, tuple[str | int, ...]]

    def to_pandas(
        self,
        select: Iterable[str],
        unstack: Iterable[AxisName] | None = None,
    ) -> pd.DataFrame:
        """Export selected fields to a pandas DataFrame.

        Fields are named by a dotted path from this result, such as
        `"functions.objectives"`, `"scaled.variables"` or `"target_objective"`.
        A path may end in one or more mapping keys, as `"metadata.run.id"` does.
        Each selected path becomes a column of that name.

        Multi-dimensional fields are stacked into rows indexed by a multi-index
        derived from the field's axis metadata; index levels are labeled using
        the `names` mapping (numeric indices if absent). `batch_id` is always
        prepended to the index. The `unstack` argument pivots selected axes into
        columns, producing tuple column names of the form `(path, label, ...)`.

        Paths whose value is `None`, and missing mapping keys, are skipped. A
        path that does not name a value raises a `ValueError`.

        See [Working with Results](../optimizer_setup/results.md#exporting-to-pandas) for
        further details and examples.

        Args:
            select:  The dotted paths of the fields to export.
            unstack: Axes to pivot into columns (default: none).

        Returns:
            A DataFrame with the selected fields as columns.

        Raises:
            UnsupportedError: If the `pandas` module is not installed.
        """
        if not HAVE_PANDAS:
            msg = missing_engine_message("pandas", "to_pandas", "use to_polars")
            raise UnsupportedError(msg)

        return _to_pandas_frame(self, select, unstack)

    def to_polars(
        self,
        select: Iterable[str],
        unstack: Iterable[AxisName] | None = None,
        sep: str = ",",
    ) -> pl.DataFrame:
        """Export selected fields to a polars DataFrame.

        This is the polars counterpart of
        [`to_pandas`][ropt.results.Results.to_pandas], returned in long
        format with tuple column names joined into a single string using
        `sep`. See [Exporting to polars](../optimizer_setup/results.md#exporting-to-polars)
        for details.

        Args:
            select:  The dotted paths of the fields to export.
            unstack: Axes to pivot into columns (default: none).
            sep:     Separator used to join unstacked column names.

        Returns:
            A DataFrame with axis labels and the selected fields as columns.

        Raises:
            UnsupportedError: If the `polars` module is not installed.
        """
        if not HAVE_POLARS:
            msg = missing_engine_message("polars", "to_polars", "use to_pandas")
            raise UnsupportedError(msg)

        return _to_polars_frame(self, select, unstack, sep)[0]
