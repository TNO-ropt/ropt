"""Gradient calculators: the methods that calculate gradients from evaluated points.

A gradient calculator is selected through the `gradient` section of the
configuration, either as an instance or as a
[`GradientConfig`][ropt.config.GradientConfig] naming a method.
[`DefaultGradientCalculator`][ropt.gradient.default.DefaultGradientCalculator]
provides the stochastic method by default and the finite-difference methods
`forward`, `backward` and `central`.

See [Stochastic Gradients](../optimizer_setup/gradients.md) and
[Finite-Difference Gradients](../optimizer_setup/finite_differences.md) for the
built-in methods, and [Writing a Plugin](../advanced/writing_plugins.md) for
implementing a calculator.
"""

from ._base import GradientCalculator, GradientData

__all__ = [
    "GradientCalculator",
    "GradientData",
]
