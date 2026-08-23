"""Plugin support for samplers.

A sampler generates the perturbed variable vectors that are used to estimate
stochastic gradients. A [`SamplerPlugin`][ropt.plugins.sampler.SamplerPlugin] is
a factory that creates the [`Sampler`][ropt.sampler.Sampler] objects doing the
actual work, which the [`PluginManager`][ropt.plugins.manager.PluginManager]
discovers through the `ropt.plugins.sampler` entry point group.

`ropt` ships [`SciPySampler`][ropt.sampler.scipy.SciPySampler], which is based
on the `scipy.stats` and `scipy.stats.qmc` packages.

See [Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

from ._base import SamplerPlugin

__all__ = [
    "SamplerPlugin",
]
