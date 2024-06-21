"""Configuration class for samplers."""

from __future__ import annotations

from typing import Any, Dict

from ._enopt_base_model import EnOptBaseModel


class SamplerConfig(EnOptBaseModel):
    """The sampler configuration class.

    This class defines the configuration for samplers, which are configured by
    the `samplers` field in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig]
    object. That field contains a tuple of configuration objects that define
    which samplers are available during the optimization. The `samplers` field
    in the gradient configuration
    ([`GradientConfig`][ropt.config.enopt.GradientConfig]) is used to specify
    the index of the sampler for each variable.

    Gradients are calculated from a set of perturbed variables, which may be
    deterministic or stochastic in nature. These perturbations are generally
    produced by sampler objects that produce perturbation values to add to the
    unperturbed variables.

    Perturbation values are produced by a sampler, which provides the methods
    that can be used. The `method` field determines which sampler method will be
    used. To further specify how such a method should function, the `options`
    field can be used to pass a dictionary of key-value pairs. The
    interpretation of these options depends on the chosen method.

    By default, a different set of perturbed variables is generated for each
    realization. By setting the `shared` flag to `True`, the sampler can be
    directed to use the same set of perturbed values for each realization.

    Attributes:
        method:  The sampler method
        options: Options to be passed to the sampler
        shared:  Whether perturbation values should be shared between realizations
                 (default: `False`)
    """

    method: str = "scipy/default"
    options: Dict[str, Any] = {}  # noqa: RUF012
    shared: bool = False
