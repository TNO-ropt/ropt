"""The objects a batch evaluation produces.

Every evaluation yields a tuple of [`Results`][ropt.results.Results]: a
[`FunctionResults`][ropt.results.FunctionResults] for objective and constraint
values, a [`GradientResults`][ropt.results.GradientResults] for gradient
estimates, or both. Each is a frozen container of
[`ResultField`][ropt.results.ResultField] sub-objects holding NumPy arrays with
axis-name metadata.

See [Working with Results](../results/results.md) for a narrative overview of
the result hierarchy, axis metadata, scaling, and pandas/polars export.
"""

from ._constraint_info import ConstraintInfo
from ._function_aggregates import FunctionAggregates
from ._function_evaluations import FunctionEvaluations
from ._function_results import FunctionResults, ScaledFunctionResults
from ._gradient_aggregates import GradientAggregates
from ._gradient_perturbations import GradientPerturbations, ScaledGradientPerturbations
from ._gradient_results import GradientResults, ScaledGradientResults
from ._pandas_frame import results_to_pandas
from ._polars_frame import results_to_polars
from ._realizations import Realizations
from ._result_field import AxisMetadata, ResultField
from ._results import Results

__all__ = [
    "AxisMetadata",
    "ConstraintInfo",
    "FunctionAggregates",
    "FunctionEvaluations",
    "FunctionResults",
    "GradientAggregates",
    "GradientPerturbations",
    "GradientResults",
    "Realizations",
    "ResultField",
    "Results",
    "ScaledFunctionResults",
    "ScaledGradientPerturbations",
    "ScaledGradientResults",
    "results_to_pandas",
    "results_to_polars",
]
