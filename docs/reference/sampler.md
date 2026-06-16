# Samplers

Samplers generate the perturbation vectors used for stochastic gradient
estimation. The default SciPy-based sampler covers normal- and
quasi-random-distribution sampling; alternative samplers can be added through
the plugin system.

See [Stochastic Gradients](../usage/gradients.md) for how samplers fit into
the gradient pipeline.

::: ropt.sampler
::: ropt.sampler.Sampler
::: ropt.sampler.scipy.SciPySampler

