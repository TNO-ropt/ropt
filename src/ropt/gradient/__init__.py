"""Gradient calculators: the methods that calculate gradients from evaluated points.

A gradient calculator is selected through the `gradient` section of the
configuration, either as an instance or as a
[`GradientConfig`][ropt.config.GradientConfig] naming a method.
[`DefaultGradientCalculator`][ropt.gradient.default.DefaultGradientCalculator]
provides the stochastic method.

See [Stochastic Gradients](../optimizer_setup/gradients.md) for the stochastic
method, and [Writing a Plugin](../advanced/writing_plugins.md) for implementing
a calculator.
"""

from ._base import GradientCalculator, GradientData

__all__ = [
    "GradientCalculator",
    "GradientData",
]
