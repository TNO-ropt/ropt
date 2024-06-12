"""Utilities used by the script-based optimizer."""

from collections import defaultdict
from itertools import groupby
from typing import Any, DefaultDict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig


def _make_dict(
    variables: NDArray[np.float64], names: Sequence[Tuple[str, ...]]
) -> DefaultDict[str, Any]:
    def recursive_dict() -> DefaultDict[str, Any]:
        return defaultdict(recursive_dict)

    var_dict = recursive_dict()
    for idx, var_name in enumerate(names):
        tmp = var_dict
        for name in var_name[:-1]:
            tmp = tmp[str(name)]
        tmp[str(var_name[-1])] = variables[idx]

    return var_dict


def _get_function_files(config: EnOptConfig) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    assert config.objective_functions.names is not None
    objective_names = tuple(str(name) for name in config.objective_functions.names)
    if config.nonlinear_constraints is not None:
        assert config.nonlinear_constraints.names is not None
        constraint_names = tuple(
            str(name) for name in config.nonlinear_constraints.names
        )
    else:
        constraint_names = ()
    return objective_names, constraint_names


def _format_list(values: List[int]) -> str:
    grouped = (
        tuple(y for _, y in x)
        for _, x in groupby(enumerate(sorted(values)), lambda x: x[0] - x[1])
    )
    return ", ".join(
        "-".join([str(sub_group[0]), str(sub_group[-1])])
        if len(sub_group) > 1
        else str(sub_group[0])
        for sub_group in grouped
    )
