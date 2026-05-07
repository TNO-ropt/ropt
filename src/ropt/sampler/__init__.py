"""Public API for sampler implementations.

Samplers generate perturbation values for optimization variables during gradient
estimation. In ensemble-based workflows, these perturbations are used to
evaluate objective and constraint functions under variable perturbations.

**Core Interface**

All sampler implementations inherit from the [`Sampler`][ropt.sampler.Sampler]
base class, which defines the sampler lifecycle (`__init__`, `init`) and the
required sample generation method (`generate_samples`).

**Integration with Optimization**

Samplers are accessed via an [`EnOptContext`][ropt.context.EnOptContext] object
through its `samplers` field, a tuple of sampler instances. Samplers are
instantiated either directly as objects or via
[`SamplerConfig`][ropt.config.SamplerConfig] objects, which are used by the
plugin system to create instances based on the configured method string (e.g.,
`"default"` or `"sobol"`).

**Built-in and Custom Samplers**

The [`SciPySampler`][ropt.sampler.scipy.SciPySampler] class provides sampling
methods backed by `scipy.stats` and `scipy.stats.qmc`.

Users can implement custom samplers by subclassing `Sampler`. Those subclasses
can be instantiated directly and passed into an
[`EnOptContext`][ropt.context.EnOptContext] object through its `samplers` field.
Registering a custom sampler with the plugin system is optional and only
required when the sampler should be selected and configured via `SamplerConfig`
objects instead of being instantiated explicitly by the user.
"""

from ._base import Sampler

__all__ = [
    "Sampler",
]
