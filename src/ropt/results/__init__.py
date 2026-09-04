"""Data classes for storing intermediate optimization results.

See [Working with Results](../optimizer_setup/results.md) for a narrative overview of
the result hierarchy, axis metadata, scaling, and pandas/polars export.
"""

from ._constraint_info import ConstraintInfo
from ._function_evaluations import FunctionEvaluations
from ._function_results import FunctionResults
from ._functions import Functions
from ._gradient_evaluations import GradientEvaluations
from ._gradient_results import GradientResults
from ._gradients import Gradients
from ._pandas_frame import results_to_pandas
from ._polars_frame import results_to_polars
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
    "results_to_pandas",
    "results_to_polars",
]
