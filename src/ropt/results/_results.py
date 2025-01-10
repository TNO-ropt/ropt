from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from importlib.util import find_spec
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Iterable,
    TypeVar,
)

if TYPE_CHECKING:
    from ropt.enums import ResultAxis


_HAVE_PANDAS: Final = find_spec("pandas") is not None

if TYPE_CHECKING and _HAVE_PANDAS:
    import pandas as pd  # noqa: TC002
if _HAVE_PANDAS:
    from ._pandas import _to_dataframe

TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class Results(ABC):
    """The `Results` class serves as an abstract base class for storing results.

    This class contains the following generic information:

    1. The plan ID, a tuple of integer values.
    2. The result ID, a unique integer value.
    3. An optional batch ID, which may be generated by the function evaluator.
       The interpretation of this ID depends on the evaluator code. It is
       intended to be used as a unique identifier for the group of function
       evaluations passed to the evaluator by the optimization code.
    4. A dictionary of metadata to be added by optimization steps. This contains
       generic information, the nature of which depends on the steps producing
       them. They are expected to be primitive values not interpreted by the
       optimization code but can be exported and reported.

    The `Results` class is an abstract base class that is not intended to be
    instantiated by itself. Most data of interest will be stored in additional
    fields in one of the derived classes:
    [`FunctionResults`][ropt.results.FunctionResults] or
    [`GradientResults`][ropt.results.GradientResults]. In addition to the
    attributes containing the data, a few methods are provided to export the
    data. These functions will only be useful when used with one of the derived
    classes:

    1. The [`to_dataframe`][ropt.results.Results.to_dataframe] method can be
       used to export the contents, or a sub-set, of a single field to a
       [`pandas`](https://pandas.pydata.org/) data frame.

    Attributes:
        plan_id:   The plan ID.
        result_id: The ID of the function/gradient evaluation.
        batch_id:  The ID of the evaluation batch that contains the result.
        metadata:  The metadata.
    """

    plan_id: tuple[int, ...]
    result_id: int | tuple[int, ...]
    batch_id: int | None
    metadata: dict[str, Any]

    def to_dataframe(
        self,
        field_name: str,
        select: Iterable[str] | None = None,
        unstack: Iterable[ResultAxis] | None = None,
        names: dict[ResultAxis, tuple[str, ...] | None] | None = None,
    ) -> pd.DataFrame:
        """Export a field to a pandas dataframe.

        The function exports the values of a single field to a pandas data
        frame. The field to export is selected by the `field_name` argument. In
        general such a field is another object with multiple sub-fields. By
        default, these are all exported as columns in the pandas data frame, but
        a sub-set can be selected using the `select` argument.

        Any of the sub-fields in the field that is exported may be a
        multi-dimensional array, which is exported in a stacked manner. Using
        the axis types found in the metadata, the exporter will construct a
        multi-index labeled with the corresponding names provided via the
        `names` argument. If `names` is `None`, numerical indices are used.
        Such multi-indices can optionally be unstacked into multiple columns
        by providing the axis types to unstack via the `unstack` argument.

        Info: The data frame index
            As noted above, the index of the resulting data frame may be a
            multi-index constructed from axis indices or labels. In addition,
            the `plan_id`, `result_id`, and the `batch_id` (if not None) fields,
            are also prepended to the index of the resulting frame. This has the
            beneficial effect that the data frames exported from multiple
            results can be concatenated and identified in the frame by plan ID,
            result ID and/or batch ID.

        Args:
            field_name: The field to export.
            select:     Select the sub-fields to export, by default all fields.
            unstack:    Select axes to unstack, by default none.
            names:      A dictionary mapping axis types to names.

        Raises:
            NotImplementedError: If the pandas module is not installed.
            ValueError:          If the field name is incorrect.

        Returns:
            A pandas data frame containing the results.

        Warning:
            This function is only available if `pandas` is installed.
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
            self.plan_id,
            self.result_id,
            self.batch_id,
            select,
            unstack,
            names,
        )
