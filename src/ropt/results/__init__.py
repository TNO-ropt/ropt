"""The objects a batch evaluation produces.

Every evaluation yields a tuple of [`Results`][ropt.results.Results]: a
[`FunctionResults`][ropt.results.FunctionResults] for objective and constraint
values, a [`GradientResults`][ropt.results.GradientResults] for gradient
estimates, or both. Each is a frozen container of
[`ResultField`][ropt.results.ResultField] sub-objects holding NumPy arrays with
axis-name metadata.

See [Working with Results](../running/results.md) for a narrative overview of
the result hierarchy, axis metadata, scaling, and pandas/polars export.
"""

from ._constraint_info import ConstraintInfo
from ._function_evaluations import FunctionEvaluations
from ._function_results import FunctionResults, ScaledFunctionResults
from ._functions import Functions
from ._gradient_evaluations import GradientEvaluations
from ._gradient_results import GradientResults, ScaledGradientResults
from ._gradients import Gradients
from ._pandas_frame import results_to_pandas
from ._polars_frame import results_to_polars
from ._realizations import Realizations
from ._result_field import AxisMetadata, ResultField
from ._results import Results

__all__ = [
    "AxisMetadata",
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
    "ScaledFunctionResults",
    "ScaledGradientResults",
    "results_to_pandas",
    "results_to_polars",
]
