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
(<AxisName.REALIZATION: 'realization'>, <AxisName.OBJECTIVE: 'objective'>)
```
Given that the first axis denotes realizations and the second axis denotes
objectives, each row in the array represents the set of objective values for a
specific realization. This metadata provides the necessary context for exporting
and reporting code to associate each element in the result matrix with its
corresponding realization and objective, as specified in the optimizer
configuration. The pandas exporting code, for example, utilizes this information
to construct a multi-index for the output DataFrame and to transform the
multi-dimensional data into multiple columns.

The [`AxisName`][ropt.enums.AxisName] enumeration currently defines the
following axes:

- [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE] The index along this
  axis refers to the objective number as specified in the
  [`EnOptConfig`][ropt.config.enopt.EnOptConfig] configuration.
- [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]
  The index along this axis corresponds to the non-linear constraint index
  defined in the [`EnOptConfig`][ropt.config.enopt.EnOptConfig] configuration.
- [`AxisName.LINEAR_CONSTRAINT`][ropt.enums.AxisName.LINEAR_CONSTRAINT] The
  index along this axis corresponds to the linear constraint index defined in
  the [`EnOptConfig`][ropt.config.enopt.EnOptConfig] configuration.
- [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE] The index along this
  axis refers to the variable number as specified by the
  [`EnOptConfig`][ropt.config.enopt.EnOptConfig] configuration.
- [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]: When results
  involve an ensemble, this axis represents the different realizations, where
  the index corresponds to the realization number.
- [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION] For gradient
  calculations, multiple variable perturbations are used. The objectives and
  constraints calculated for each perturbation are reported along this axis,
  which represents the perturbation index.

Refer to the documentation of the individual result classes for the exact
dimensions of each result field. The dimensionality of the data and the order of
the axes are fixed and listed sequentially for every field.

Note:
    The dimensionality associated with a result axis is fixed. For instance,
    even with only a single objective, results containing objective values will
    still include a [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]
    axis of length one.
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
