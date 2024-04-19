"""Configuration class for samplers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._enopt_base_model import EnOptBaseModel
from .constants import DEFAULT_SAMPLER_BACKEND


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

    Perturbation values are produced by a sampler backend, which provides the
    methods that can be used. The `backend` field is used to select the backend,
    which may be either built-in or installed separately as a plugin. A backend
    may implement multiple algorithms, and the `method` field determines which
    one will be used. To further specify how such a method should function, the
    `options` field can be used to pass a dictionary of key-value pairs. The
    interpretation of these options depends on the backend and the chosen
    method.

    By default, a different set of perturbed variables is generated for each
    realization. By setting the `shared` flag to `True`, the sampler can be
    directed to use the same set of perturbed values for each realization.

    Attributes:
        backend: The name of the sampler backend (default:
            [`DEFAULT_SAMPLER_BACKEND`][ropt.config.enopt.constants.DEFAULT_SAMPLER_BACKEND])
        method:  The sampler method
        options: Options to be passed to the sampler
        shared:  Whether perturbation values should be shared between realizations
                (default: `False`)
    """

    backend: str = DEFAULT_SAMPLER_BACKEND
    method: Optional[str] = None
    options: Dict[str, Any] = {}  # noqa: RUF012
    shared: bool = False
