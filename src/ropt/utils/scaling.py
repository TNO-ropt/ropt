"""Functions for scaling variables and functions."""

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig


def scale_objectives(
    config: EnOptConfig,
    objectives: NDArray[np.float64],
    scales: NDArray[np.float64] | None,
    axis: int,
) -> NDArray[np.float64] | None:
    """Scale objective function values.

    Given an ensemble optimizer configuration object, this function scales the
    provided objective values. It divides them by the scale values given in the
    configuration object (if not `None`), and optionally also by the values
    given in the `scales` argument.

    As the `objectives` input may be a multi-dimensional array, the index of the
    axis that designates the objectives should be provided through the `axis`
    argument.

    Args:
        config:     The ensemble optimizer configuration object.
        objectives: Objective functions.
        scales:     Optional additional scales.
        axis:       The objectives axis.

    Returns:
        The scaled objectives or `None` if no scaling was applied.
    """
    total_scales = config.objectives.scales
    if scales is not None:
        total_scales = total_scales * scales
    if np.allclose(total_scales, 1.0, rtol=0.0, atol=1e-10):
        return None
    objectives = np.moveaxis(objectives, axis, -1)
    return np.moveaxis(objectives / total_scales, -1, axis)


def scale_constraints(
    config: EnOptConfig,
    constraints: NDArray[np.float64] | None,
    scales: NDArray[np.float64] | None,
    axis: int,
) -> NDArray[np.float64] | None:
    """Scale constraint function values.

    Given an ensemble optimizer configuration object, this function scales the
    provided constraint values. It divides them by the scale values given in the
    configuration object (if not `None`), and optionally also by the values
    given in the `scales` argument.

    As the `constraints` input may be a multi-dimensional array, the index of the
    axis that designates the constraints should be provided through the `axis`
    argument.

    Args:
        config:      The ensemble optimizer configuration object.
        constraints: Constraint functions.
        scales:      Optional additional scales.
        axis:        The constraints axis.

    Returns:
        The scaled constraints or `None` if no scaling was applied.
    """
    if constraints is None or config.nonlinear_constraints is None:
        return None
    total_scales = config.nonlinear_constraints.scales
    if scales is not None:
        total_scales = total_scales * scales
    if np.allclose(total_scales, 1.0, rtol=0.0, atol=1e-10):
        return None
    constraints = np.moveaxis(constraints, axis, -1)
    return np.moveaxis(constraints / total_scales, -1, axis)
