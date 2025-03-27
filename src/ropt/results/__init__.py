"""Data classes for storing intermediate optimization results.

During the optimization process, the calculation of functions and gradients
generates data that needs to be reported. To facilitate this, new results are
passed to callbacks as a sequence of [`Results`][ropt.results.Results] objects.
These objects can be instances of either the
[`FunctionResults`][ropt.results.FunctionResults] or
[`GradientResults`][ropt.results.GradientResults] classes, which store the
results of function and gradient evaluations, respectively.

Much of the data within these result objects is multi-dimensional. For example,
the `objectives` field, which is part of the nested
[`evaluations`][ropt.results.FunctionEvaluations] object within
[`FunctionResults`][ropt.results.FunctionResults], is a two-dimensional `numpy`
array. In this array, each column represents a different objective, and each row
corresponds to a specific realization number.

To simplify exporting and reporting, the identity of the axes in these
multi-dimensional arrays is stored as metadata associated with each field. These
fields are derived from the [`ResultField`][ropt.results.ResultField] class,
which provides a [`get_axes`][ropt.results.ResultField.get_axes] class method
for retrieving the axes. For instance, for the `objectives` field, this method
would return:

```py
>>> from ropt.results import FunctionEvaluations
>>> FunctionEvaluations.get_axes("objectives")
(<ResultAxis.REALIZATION: 'realization'>, <ResultAxis.OBJECTIVE: 'objective'>)
```
Given that the first axis denotes realizations and the second axis denotes
objectives, each row in the array represents the set of objective values for a
specific realization. This metadata provides the necessary context for exporting
and reporting code to associate each element in the result matrix with its
corresponding realization and objective, as specified in the optimizer
configuration. The pandas exporting code, for example, utilizes this information
to construct a multi-index for the output DataFrame and to transform the
multi-dimensional data into multiple columns.
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
