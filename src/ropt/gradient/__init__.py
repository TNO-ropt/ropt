"""Gradient estimators: the methods that estimate gradients from evaluated points.

A gradient estimator is selected through the `gradient` section of the
configuration, either as an instance or as a
[`GradientConfig`][ropt.config.GradientConfig] naming a method.
[`DefaultGradientEstimator`][ropt.gradient.default.DefaultGradientEstimator]
provides the stochastic method.

See [Stochastic Gradients](../optimizer_setup/gradients.md) for the stochastic
method, and [Writing a Plugin](../advanced/writing_plugins.md) for implementing
an estimator.
"""

from ._base import GradientData, GradientEstimator

__all__ = [
    "GradientData",
    "GradientEstimator",
]
