from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from ropt.exceptions import UnsupportedError

from ._frame_support import (
    HAVE_PANDAS,
    HAVE_POLARS,
    missing_engine_message,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ropt.context import EnOptContext
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
class Results(ABC):
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
        field_name: str,
        select: Iterable[str],
        unstack: Iterable[AxisName] | None = None,
    ) -> pd.DataFrame:
        """Export a field to a pandas DataFrame.

        Exports the sub-fields of `field_name` as columns, named after the
        sub-field. Multi-dimensional sub-fields are stacked into rows indexed
        by a multi-index derived from the field's axis metadata; index levels
        are labeled using the `names` mapping (numeric indices if absent).
        `batch_id` is always prepended to the index. The `unstack`
        argument pivots selected axes into columns, producing tuple column
        names of the form `(sub-field, label, ...)`.

        See [Working with Results](../optimizer_setup/results.md#exporting-to-pandas) for
        further details and examples.

        Args:
            field_name: The field to export.
            select:     Sub-fields to include.
            unstack:    Axes to pivot into columns (default: none).

        Returns:
            A DataFrame with sub-fields as columns and axis indices as rows.

        Raises:
            UnsupportedError: If the `pandas` module is not installed.
            AttributeError:      If the field name is invalid.
        """
        if not HAVE_PANDAS:
            msg = missing_engine_message("pandas", "to_pandas", "use to_polars")
            raise UnsupportedError(msg)

        if getattr(self, field_name, None) is None:
            msg = f"Invalid result field: {field_name}"
            raise AttributeError(msg)

        return _to_pandas_frame(self, field_name, select, unstack)

    def to_polars(
        self,
        field_name: str,
        select: Iterable[str],
        unstack: Iterable[AxisName] | None = None,
        sep: str = ",",
    ) -> pl.DataFrame:
        """Export a field to a polars DataFrame.

        This is the polars counterpart of
        [`to_pandas`][ropt.results.Results.to_pandas], returned in long
        format with tuple column names joined into a single string using
        `sep`. See [Exporting to polars](../optimizer_setup/results.md#exporting-to-polars)
        for details.

        Args:
            field_name: The field to export.
            select:     Sub-fields to include.
            unstack:    Axes to pivot into columns (default: none).
            sep:        Separator used to join unstacked column names.

        Returns:
            A DataFrame with axis labels and sub-fields as columns.

        Raises:
            UnsupportedError: If the `polars` module is not installed.
            AttributeError:      If the field name is invalid.
        """
        if not HAVE_POLARS:
            msg = missing_engine_message("polars", "to_polars", "use to_pandas")
            raise UnsupportedError(msg)

        if getattr(self, field_name, None) is None:
            msg = f"Invalid result field: {field_name}"
            raise AttributeError(msg)

        return _to_polars_frame(self, field_name, select, unstack, sep)[0]

    @abstractmethod
    def unscale(self, context: EnOptContext) -> Results:
        """Unscale the results.

        Restores the quantities as configured: values are multiplied by their
        scales and offsets are added back, and objectives and gradients are
        negated again where `maximize` is set.

        Args:
            context: The context used by the source of the results.

        Returns:
            A new, unscaled `Results` object.
        """
