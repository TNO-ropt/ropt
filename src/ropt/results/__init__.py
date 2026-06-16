"""Data classes for storing intermediate optimization results.

See [Working with Results](../usage/results.md) for a narrative overview of
the result hierarchy, axis metadata, domain transforms, and pandas export.
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
from ._utils import DomainType

__all__ = [
    "ConstraintInfo",
    "DomainType",
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
