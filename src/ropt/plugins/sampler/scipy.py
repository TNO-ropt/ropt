"""This module implements the SciPy sampler plugin."""

import copy
import warnings
from typing import Any, Final

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.stats import norm, rv_continuous, truncnorm, uniform
from scipy.stats.qmc import Halton, LatinHypercube, QMCEngine, Sobol, scale

from ropt.config.enopt import EnOptConfig

from .base import Sampler, SamplerPlugin

_STATS_SAMPLERS: Final[dict[str, Any]] = {
    "uniform": uniform,
    "norm": norm,
    "truncnorm": truncnorm,
}

_QMC_ENGINES: Final = {
    "sobol": Sobol,
    "halton": Halton,
    "lhs": LatinHypercube,
}

_SUPPORTED_METHODS: Final[set[str]] = set(_STATS_SAMPLERS.keys()) | set(
    _QMC_ENGINES.keys()
)


class SciPySampler(Sampler):
    """A sampler implementation utilizing SciPy's statistical functions.

    This sampler leverages functions from the `scipy.stats` and
    `scipy.stats.qmc` modules to generate perturbation samples for optimization.

    **Supported Sampling Methods:**

    - **From Probability Distributions
      ([`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)):**
        - `norm`: Samples from a standard normal distribution (mean 0,
          standard deviation 1). This is the **default** method if none is
          specified or if "default" is requested.
        - `truncnorm`: Samples from a truncated normal distribution (mean 0,
          std dev 1), truncated to the range `[-1, 1]` by default.
        - `uniform`: Samples from a uniform distribution. Defaults to the
          range `[-1, 1]`.

    - **From Quasi-Monte Carlo Sequences
      ([`scipy.stats.qmc`](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html)):**
        - `sobol`: Uses Sobol' sequences.
        - `halton`: Uses Halton sequences.
        - `lhs`: Uses Latin Hypercube Sampling.
        *(Note: QMC samples are generated in the unit hypercube `[0, 1]^d` and
        then scaled to the hypercube `[-1, 1]^d`)*

    **Configuration:**

    The specific sampling method is chosen via the `method` field in the
    [`SamplerConfig`][ropt.config.enopt.SamplerConfig]. Additional
    method-specific parameters (e.g., distribution parameters like `loc`,
    `scale`, `a`, `b` for `stats` methods, or engine parameters for `qmc`
    methods) can be passed through the `options` dictionary within the
    `SamplerConfig`. Refer to the
    [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) and
    documentation for available options.
    """

    def __init__(
        self,
        enopt_config: EnOptConfig,
        sampler_index: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> None:
        """Initialize the sampler object.

        See the [ropt.plugins.sampler.base.Sample][] abstract base class.

        # noqa
        """
        self._enopt_config = enopt_config
        self._sampler_config = enopt_config.samplers[sampler_index]
        self._mask = mask
        _, _, self._method = self._sampler_config.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "norm"
        self._rng = rng
        self._sampler: rv_continuous | QMCEngine
        self._options: dict[str, Any]
        if self._method not in _SUPPORTED_METHODS:
            msg = f"Method `{self._method}` is not implemented by the SciPy plugin"
            raise NotImplementedError(msg)
        self._sampler, self._options = self._init_sampler(self._sampler_config.options)

    def generate_samples(self) -> NDArray[np.float64]:
        """Generate a set of samples.

        See the [ropt.plugins.sampler.base.Sampler][] abstract base class.

        # noqa
        """
        variable_count = self._enopt_config.variables.initial_values.size
        realization_count = self._enopt_config.realizations.weights.size
        perturbation_count = self._enopt_config.gradient.number_of_perturbations

        sample_dim = variable_count if self._mask is None else self._mask.sum()

        if self._method in _STATS_SAMPLERS:
            samples = self._generate_stats_samples(
                1 if self._sampler_config.shared else realization_count,
                perturbation_count,
                sample_dim,
            )
        else:
            samples = self._generate_qmc_samples(
                1 if self._sampler_config.shared else realization_count,
                perturbation_count,
                sample_dim,
            )

        if self._sampler_config.shared:
            samples = np.repeat(samples, realization_count, axis=0)

        if self._mask is not None:
            shape = (realization_count, perturbation_count, variable_count)
            result = np.zeros(shape, dtype=np.float64)
            result[..., self._mask] = samples
            return result
        return samples

    def _init_sampler(
        self, options: dict[str, Any]
    ) -> tuple[rv_continuous | QMCEngine, dict[str, Any]]:
        options = copy.deepcopy(options)
        if self._method in _STATS_SAMPLERS:
            self._set_options(options)
            sampler = _STATS_SAMPLERS[self._method]
        elif self._method in _QMC_ENGINES:
            sample_dim = (
                self._enopt_config.variables.initial_values.size
                if self._mask is None
                else self._mask.sum()
            )
            sampler = _QMC_ENGINES[self._method](sample_dim, seed=self._rng, **options)
        else:
            msg = "sampler {self._method} is not supported by this SciPy version"
            raise NotImplementedError(msg)
        return sampler, options

    def _set_options(self, options: dict[str, Any]) -> None:
        parameters = {
            "uniform": {"loc": -1.0, "scale": 2.0},
            "truncnorm": {"a": -1.0, "b": 1.0},
        }
        if self._method in parameters:
            for key, value in parameters[self._method].items():
                options.setdefault(key, value)

    def _generate_stats_samples(
        self, realization_count: int, perturbation_count: int, sample_dim: int
    ) -> NDArray[np.float64]:
        return np.array(
            self._sampler.rvs(
                size=(realization_count, perturbation_count, sample_dim),
                random_state=self._rng,
                **self._options,
            ),
        )

    def _generate_qmc_samples(
        self, realization_count: int, perturbation_count: int, sample_dim: int
    ) -> NDArray[np.float64]:
        def _run_qmc_engine() -> NDArray[np.float64]:
            return np.array(
                scale(
                    self._sampler.random(realization_count * perturbation_count),
                    np.repeat(-1.0, sample_dim),
                    np.repeat(1.0, sample_dim),
                ).T.reshape((realization_count, perturbation_count, sample_dim)),
            )

        if self._method == "sobol":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _run_qmc_engine()
        else:
            return _run_qmc_engine()


class SciPySamplerPlugin(SamplerPlugin):
    """Default sampler plugin class."""

    @classmethod
    def create(
        cls,
        enopt_config: EnOptConfig,
        sampler_index: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> SciPySampler:
        """Initialize the sampler plugin.

        See the [ropt.plugins.sampler.base.SamplerPlugin][] abstract base class.

        # noqa
        """
        return SciPySampler(enopt_config, sampler_index, mask, rng)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})
