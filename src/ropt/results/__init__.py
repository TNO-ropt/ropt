"""Data Classes for Storing Intermediate Optimization Results.

During the optimization process, functions and gradients are calculated and need
to be reported. To streamline this process, events are triggered after new
results become available, invoking connected callbacks with those results.
The [`OptimizationEvent`][ropt.events.OptimizationEvent] object passed to the
callbacks contains a `results` field, which consists of a tuple of
[`Results`][ropt.results.Results] objects. These may be instances of the derived
[`FunctionResults`][ropt.results.FunctionResults] or
[`GradientResults`][ropt.results.GradientResults] classes, containing results
for function and gradient evaluations, respectively.

These classes are nested `dataclasses` with some added features:

- Variables may have scaling and offset factors associated with them. The
  optimization will generally deal with scaled values. When the results object
  is created, and variable scaling is configured, the values are scaled back and
  added to the results object.
- Objective and constraint function values may also have a scaling factor
  associated with them. These are generally handled in unscaled form, and when
  the report object is created, scaled values are added.
- Methods are available to export the results to
  [`pandas`](https://pandas.pydata.org/) data frames or
  [`xarray`](https://xarray.dev/) data sets.
- Methods are added to write and read netCDF version 4 files using the
  [`netcdf4`](https://unidata.github.io/netcdf4-python/) Python package.

Much of the data stored in the result objects is of a multi-dimensional nature.
For instance, the `objectives` field, which is part of the nested
[`evaluations`][ropt.results.FunctionEvaluations] object in the
[`FunctionResults`][ropt.results.FunctionResults] object, is a two-dimensional
`numpy` array. In this array, the columns correspond to the objectives, and the
rows correspond to the realization number.

To facilitate exporting and reporting the results, the identity of the axes in
such multi-dimensional arrays is stored in metadata associated with the
corresponding field. These fields derive from the
[`ResultField`][ropt.results.ResultField] class, which has a
[`get_axis_names`][ropt.results.ResultField.get_axis_names] class method to
retrieve the names. For the `objectives` example above, this retrieves the axis
names:

```py
>>> from ropt.results import FunctionEvaluations
>>> FunctionEvaluations.get_axis_names("objectives")
(<ResultAxisName.REALIZATION: 'realization'>, <ResultAxisName.OBJECTIVE: 'objective'>)
```

Using this metadata, exporting or reporting code can refer to the optimizer
configuration to associate realization and objective names with any entry in the
result matrix. For instance, the [`pandas`](https://pandas.pydata.org/)
exporting code will use this to construct a multi-index for the generated data
frame and optionally unstack such multi-dimensional data into multiple columns.
"""

from ._bound_constraints import BoundConstraints
from ._function_evaluations import FunctionEvaluations
from ._function_results import FunctionResults
from ._functions import Functions
from ._gradient_evaluations import GradientEvaluations
from ._gradient_results import GradientResults
from ._gradients import Gradients
from ._linear_constraints import LinearConstraints
from ._maximize import convert_to_maximize
from ._nonlinear_constraints import NonlinearConstraints
from ._realizations import Realizations
from ._result_field import ResultField
from ._results import Results

__all__ = [
    "BoundConstraints",
    "FunctionEvaluations",
    "FunctionResults",
    "Functions",
    "GradientEvaluations",
    "GradientResults",
    "Gradients",
    "LinearConstraints",
    "NonlinearConstraints",
    "Realizations",
    "Results",
    "ResultField",
    "convert_to_maximize",
]
