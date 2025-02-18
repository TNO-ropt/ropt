"""Classes for reporting optimization results.

The classes in this module are designed to gather multiple results generated
during the execution of an optimization plan. Currently, the results can
be reported as a [`pandas`](https://pandas.pydata.org/) DataFrame using the
[`ResultsDataFrame`][ropt.report.ResultsDataFrame] class.
"""

from ._data_frame import ResultsDataFrame

__all__ = [
    "ResultsDataFrame",
]
