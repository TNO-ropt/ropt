from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ropt.context import EnOptContext
    from ropt.enums import AxisName


_HAVE_PANDAS: Final = find_spec("pandas") is not None

if TYPE_CHECKING and _HAVE_PANDAS:
    import pandas as pd  # noqa: TC002
if _HAVE_PANDAS:
    from ._pandas import _to_dataframe

TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class Results(ABC):
    """Abstract base class for optimization results.

    Subclassed by [`FunctionResults`][ropt.results.FunctionResults] and
    [`GradientResults`][ropt.results.GradientResults].

    See [Working with Results](../usage/results.md) for a narrative overview.

    Attributes:
        batch_id: Identifier for the evaluation batch.
        metadata: Dictionary of additional information (not used internally).
        names:    Mapping from [`AxisName`][ropt.enums.AxisName] to label tuples
                  for DataFrame export.
    """

    batch_id: int
    metadata: dict[str, Any]
    names: dict[str, tuple[str | int, ...]]

    def to_dataframe(
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

        See [Working with Results](../usage/results.md#exporting-to-pandas) for
        further details and examples.

        Args:
            field_name: The field to export.
            select:     Sub-fields to include.
            unstack:    Axes to pivot into columns (default: none).

        Returns:
            A DataFrame with sub-fields as columns and axis indices as rows.

        Raises:
            NotImplementedError: If the `pandas` module is not installed.
            AttributeError:      If the field name is invalid.
        """
        if not _HAVE_PANDAS:
            msg = "The pandas module must be installed to use to_dataframe"
            raise NotImplementedError(msg)

        result_field = getattr(self, field_name, None)
        if result_field is None:
            msg = f"Invalid result field: {field_name}"
            raise AttributeError(msg)

        return _to_dataframe(
            result_field,
            self.batch_id,
            select,
            unstack,
            self.names,
        )

    @abstractmethod
    def transform_from_optimizer(self, context: EnOptContext) -> Results:
        """Transform results from the optimizer domain to the user domain.

        Reverses variable, objective, and constraint transforms applied during
        optimization, restoring values to the user-defined domain.

        Args:
            context: The context used by the source of the results.

        Returns:
            A new `Results` object in the user domain.
        """
