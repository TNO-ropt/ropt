"""Data classes for storing intermediate optimization results.

During the optimization process, functions and gradients are calculated and need
to be reported. To streamline this process, new results are passed to callbacks
as a tuple of [`Results`][ropt.results.Results] objects. These may be instances
of the derived [`FunctionResults`][ropt.results.FunctionResults] or
[`GradientResults`][ropt.results.GradientResults] classes, which contain results
for function and gradient evaluations, respectively.

Much of the data stored in the result objects is multi-dimensional. For example,
the `objectives` field, which is part of the nested
[`evaluations`][ropt.results.FunctionEvaluations] object in the
[`FunctionResults`][ropt.results.FunctionResults], is a two-dimensional `numpy`
array. In this array, columns correspond to the objectives, and rows correspond
to the realization numbers.

To facilitate exporting and reporting results, the identity of the axes in such
multi-dimensional arrays is stored in metadata associated with the corresponding
field. These fields derive from the [`ResultField`][ropt.results.ResultField]
class, which includes a [`get_axes`][ropt.results.ResultField.get_axes] class
method to retrieve the axes. For example, for the `objectives` field, this
method retrieves the axes:

```py
>>> from ropt.results import FunctionEvaluations
>>> FunctionEvaluations.get_axes("objectives")
(<ResultAxis.REALIZATION: 'realization'>, <ResultAxis.OBJECTIVE: 'objective'>)
```

Using this metadata, the exporting or reporting code can refer to the optimizer
configuration to associate realization and objective names with any entry in the
result matrix. For instance, the [`pandas`](https://pandas.pydata.org/)
exporting code will utilize this information to construct a multi-index for the
generated DataFrame and may also unstack such multi-dimensional data into
multiple columns.
"""

from ._constraint_info import ConstraintInfo
from ._data_frame import results_to_dataframe
from ._function_evaluations import FunctionEvaluations
from ._function_results import FunctionResults
from ._functions import Functions
from ._gradient_evaluations import GradientEvaluations
from ._gradient_results import GradientResults
from ._gradients import Gradients
from ._realizations import Realizations
from ._result_field import ResultField
from ._results import Results

__all__ = [
    "ConstraintInfo",
    "FunctionEvaluations",
    "FunctionResults",
    "Functions",
    "GradientEvaluations",
    "GradientResults",
    "Gradients",
    "Realizations",
    "ResultField",
    "Results",
    "results_to_dataframe",
]
