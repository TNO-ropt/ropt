"""Samplers: generators of the perturbation values used for gradient estimation.

A sampler is selected through the `samplers` field of an
[`EnOptContext`][ropt.context.EnOptContext], either as an instance or as a
[`SamplerConfig`][ropt.config.SamplerConfig] naming a method.
[`SciPySampler`][ropt.sampler.scipy.SciPySampler] provides the methods backed by
`scipy.stats` and `scipy.stats.qmc`.

See [Stochastic Gradients](../optimizer_setup/gradients.md) for how sampling
fits into the gradient pipeline, and
[Writing a Plugin](../advanced/writing_plugins.md) for implementing one.
"""

from ._base import Sampler

__all__ = [
    "Sampler",
]
