"""Functions for scaling variables and functions."""

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig


def scale_variables(
    config: EnOptConfig, variables: NDArray[np.float64], axis: int
) -> NDArray[np.float64] | None:
    """Scale variables.

    Given an ensemble optimizer configuration object and a vector of variables,
    this function applies an offset and scale to the vector values.

    As the `variables` input may be a multi-dimensional array, the index of the
    axis that designates the variables should be provided through the `axis`
    argument.

    Args:
        config:    The ensemble optimizer configuration object.
        variables: The scaled variables.
        axis:      The variables axis.

    Returns:
        The scaled variables, or `None` if no scaling is applied.
    """
    if config.variables.offsets is None and config.variables.scales is None:
        return None
    variables = np.moveaxis(variables, axis, -1)
    if config.variables.offsets is not None:
        variables = variables - config.variables.offsets
    if config.variables.scales is not None:
        variables = variables / config.variables.scales
    return np.moveaxis(variables, -1, axis)


def scale_back_variables(
    config: EnOptConfig,
    variables: NDArray[np.float64],
    axis: int,
    *,
    correct_offsets: bool = True,
) -> NDArray[np.float64] | None:
    """Scale back variables.

    Given an ensemble optimizer configuration object and a vector of scaled
    variables, scale their values back to the original range. Normally this
    includes correcting for offsets, but if a difference value is being
    rescaled, the `correct_offsets` flag can be used to disable this.

    As the `variables` input may be a multi-dimensional array, the index of the
    axis that designates the variables should be provided through the `axis`
    argument.

    Args:
        config:          The ensemble optimizer configuration object.
        variables:       The scaled variables.
        axis:            The variables axis.
        correct_offsets: If True also correct for offsets.

    Returns:
        The unscaled variables, or `None` if no scaling is applied.
    """
    if config.variables.offsets is None and config.variables.scales is None:
        return None
    variables = np.moveaxis(variables, axis, -1)
    if config.variables.scales is not None:
        variables = variables * config.variables.scales
    if correct_offsets and config.variables.offsets is not None:
        variables = variables + config.variables.offsets
    return np.moveaxis(variables, -1, axis)


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
