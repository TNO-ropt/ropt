from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, Optional, Type, TypeVar, Union

from ._bound_constraints import BoundConstraints
from ._function_evaluations import FunctionEvaluations
from ._functions import Functions
from ._linear_constraints import LinearConstraints
from ._nonlinear_constraints import NonlinearConstraints
from ._realizations import Realizations

if TYPE_CHECKING:
    from ._result_field import ResultField

from ._results import Results

_HAVE_XARRAY: Final = find_spec("xarray") is not None
_HAVE_NETCDF: Final = find_spec("netCDF4") is not None

if _HAVE_XARRAY:
    from ._xarray import _from_dataset
if _HAVE_XARRAY and _HAVE_NETCDF:
    from ._xarray import _from_netcdf

TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass
class FunctionResults(Results):
    """The `FunctionResults` class stores function related results.

    This class contains  the following additional information:

    1. The results of the function evaluations.
    2. The parameters of the realizations, such as weights for objectives and
       constraints, and realization failures.
    3. The calculated objective and constraint function values.
    4. Information on constraint values and violations.

    Attributes:
        evaluations:           Results of the function evaluations.
        realizations:          The calculated parameters of the realizations.
        functions:             The calculated functions.
        bound_constraints:     Bound constraints.
        linear_constraints:    Linear constraints.
        nonlinear_constraints: Nonlinear constraints.
    """

    evaluations: FunctionEvaluations
    realizations: Realizations
    functions: Optional[Functions]
    bound_constraints: Optional[BoundConstraints] = None
    linear_constraints: Optional[LinearConstraints] = None
    nonlinear_constraints: Optional[NonlinearConstraints] = None

    @classmethod
    def from_netcdf(cls, filename: Union[str, Path]) -> Results:
        """Read results from a netCDF4 file.

        Use of this method requires that the `xarray` and `netCDF4` modules are
        installed.

        The filename is assumed to have an ".nc" extension, which will be added
        if not present.

        Args:
            filename: The name or path of the file to read.

        Raises:
            NotImplementedError: If the xarray or the netCDF4 module is not installed.

        Returns:
            The loaded result.

        Warning:
            This function is only available if `xarray` and `netCDF4` are installed.
        """
        if not _HAVE_XARRAY or not _HAVE_NETCDF:
            msg = (
                "The xarray and netCDF4 modules must be installed "
                "to use Results.from_netcdf"
            )
            raise NotImplementedError(msg)

        _types: Dict[str, Type[ResultField]] = {
            "bound_constraints": BoundConstraints,
            "linear_constraints": LinearConstraints,
            "nonlinear_constraints": NonlinearConstraints,
            "evaluations": FunctionEvaluations,
            "functions": Functions,
            "realizations": Realizations,
        }

        results = _from_netcdf(Path(filename), cls)
        kwargs: Dict[str, Any] = {
            key: _types[key](**_from_dataset(value)) if key in _types else value
            for key, value in results.items()
        }
        kwargs.update({key: None for key in _types if key not in results})
        return cls(**kwargs)
