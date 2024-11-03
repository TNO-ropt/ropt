from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, Optional, Type, TypeVar, Union

from ._gradient_evaluations import GradientEvaluations
from ._gradients import Gradients
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
class GradientResults(Results):
    """The `GradientResults` class stores gradient related results.

    This contains  the following additional information:

    1. The results of the function evaluations for perturbed variables.
    2. The parameters of the realizations, such as weights for objectives and
       constraints, and realization failures.
    3. The gradients of the calculated objectives and constraints.

    Attributes:
        evaluations:  Results of the function evaluations.
        realizations: The calculated parameters of the realizations.
        gradients:    The calculated gradients.
    """

    evaluations: GradientEvaluations
    realizations: Realizations
    gradients: Optional[Gradients]

    @classmethod
    def from_netcdf(cls: Type[TypeResults], filename: Union[str, Path]) -> TypeResults:
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
            "evaluations": GradientEvaluations,
            "realizations": Realizations,
            "gradients": Gradients,
        }

        results = _from_netcdf(Path(filename), cls)
        kwargs: Dict[str, Any] = {
            key: _types[key](**_from_dataset(value)) if key in _types else value
            for key, value in results.items()
        }
        kwargs.update({key: None for key in _types if key not in results})
        return cls(**kwargs)
