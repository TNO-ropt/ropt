"""Plugin functionality for adding sampler plugins.

Sampler plugins are managed by a [`PluginManager`][ropt.plugins.PluginManager]
object, which returns classes or factory functions to create objects that
implement one or more sampling methods to produce perturbations. These objects
must adhere to the [`Sampler`][ropt.plugins.sampler.base.Sampler] abstract base
class.

Samplers can be added via the plugin manager, by default the
[`SciPySampler`][ropt.plugins.sampler.scipy.SciPySampler] plugin is installed
which provides a number of methods from the
[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) package.
"""
