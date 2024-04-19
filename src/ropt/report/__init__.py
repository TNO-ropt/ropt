"""Classes for reporting optimization results.

The classes in this module are designed to gather multiple results generated
during the execution of an optimization workflow. Currently, the results can
be reported as a [`pandas`](https://pandas.pydata.org/) DataFrame using the
[`ResultsDataFrame`][ropt.report.ResultsDataFrame] class, or as a text file in
a tabular format using the [`ResultsTable`][ropt.report.ResultsTable] class.
"""

from ._data_frame import ResultsDataFrame
from ._table import ResultsTable

__all__ = [
    "ResultsDataFrame",
    "ResultsTable",
]
