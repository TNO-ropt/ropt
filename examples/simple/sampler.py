"""A custom sampler with the high-level ``ropt.simple`` API.

A sampler draws the perturbations that `ropt` uses to estimate stochastic
gradients. This example implements ``OneAtATime``, which perturbs a single
variable per sample, the pattern of a forward finite difference. It therefore
needs exactly one perturbation per variable, and rejects any other number.

The class provides two methods, which the configuration selects by name:

- ``forward`` perturbs every variable by the same amount, so the gradient
  estimate is a plain forward difference. Perturbations do not have to be
  random.
- ``random`` stretches or shrinks each step by a random factor.

The two also differ in the ``shared`` flag, which asks a sampler to hand the
same perturbations to every realization. For ``forward`` that changes nothing,
because a unit step does not depend on the realization it belongs to; for
``random`` it decides whether the realizations get their own step sizes.

The sampler is **registered** with ``register_plugin``, which makes it available
exactly like an installed one: it is selected from the configuration by its
``"plugin/method"`` string. A sampler defined in a script or a notebook cannot
be found through an entry point, and registering is what closes that gap.
"""

from typing import Any, ClassVar

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from ropt.config import SamplerConfig
from ropt.plugins import MethodSpec, register_plugin
from ropt.sampler import Sampler
from ropt.simple import EvaluationFunctionContext, optimize

DIM = 5
UNCERTAINTY = 0.1
REALIZATIONS = 10
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


class OneAtATime(Sampler):
    """Perturb one variable per sample, as a forward difference does.

    The samples are the rows of an identity matrix, optionally scaled by a
    random factor. Because `ropt` multiplies them by the configured
    `perturbation_magnitudes`, a unit entry makes the step exactly that
    magnitude.
    """

    # The methods this plugin provides, as an installed plugin declares them.
    methods: ClassVar[MethodSpec] = {"forward", "random"}

    def __init__(self, sampler_config: SamplerConfig) -> None:
        """Create the sampler.

        Args:
            sampler_config: The sampler configuration.
        """
        _, _, self._method = sampler_config.method.lower().rpartition("/")
        self._shared = sampler_config.shared

    def init(
        self,
        *,
        realization_count: int,
        perturbation_count: int,
        variable_count: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> None:
        """Store the sample shape and check that it suits one-at-a-time steps.

        Args:
            realization_count:  The number of realizations.
            perturbation_count: The number of perturbations per realization.
            variable_count:     The total number of variables.
            mask:               The variables this sampler handles, if not all.
            rng:                The random number generator to draw from.

        Raises:
            ValueError: If there is not exactly one perturbation per variable.
        """
        self._mask = np.ones(variable_count, dtype=np.bool_) if mask is None else mask
        if perturbation_count != self._mask.sum():
            msg = "This sampler needs one perturbation per variable."
            raise ValueError(msg)
        self._realization_count = realization_count
        self._perturbation_count = perturbation_count
        self._variable_count = variable_count
        self._rng = rng

    def generate_samples(self) -> NDArray[np.float64]:
        """Generate one perturbation per variable.

        Returns:
            Samples of shape `(realizations, perturbations, variables)`.
        """
        rows = 1 if self._shared else self._realization_count
        shape = (rows, self._perturbation_count)
        factors = (
            np.ones(shape)
            if self._method == "forward"
            else self._rng.uniform(0.5, 1.5, shape)
        )
        samples = np.zeros((rows, self._perturbation_count, self._variable_count))
        # Variables this sampler does not handle must stay at zero.
        samples[..., self._mask] = (
            np.eye(self._perturbation_count) * factors[..., np.newaxis]
        )
        if self._shared:
            samples = np.repeat(samples, self._realization_count, axis=0)
        return samples


def main() -> None:
    """Run the optimization with both sampler methods and check the results."""
    register_plugin("sampler", "custom", OneAtATime)

    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    def rosenbrock(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        r = context.realization
        objective = 0.0
        for d_idx in range(DIM - 1):
            x, y = variables[d_idx : d_idx + 2]
            objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
        return float(objective)

    # Sharing is what makes the random steps identical across realizations, so
    # only the second run gives each realization its own.
    for method, shared in (("forward", True), ("random", False)):
        config: dict[str, Any] = {
            "variables": {
                "variable_count": DIM,
                "perturbation_magnitudes": 1e-6,
            },
            "realizations": {
                "weights": [1.0] * REALIZATIONS,
            },
            # One perturbation per variable is what this sampler requires.
            "gradient": {
                "number_of_perturbations": DIM,
            },
            "samplers": [
                {"method": f"custom/{method}", "shared": shared},
            ],
        }

        result = optimize(config, INITIAL_VALUES, rosenbrock)
        print(f"{method} (shared={shared}):")
        assert result.results is not None
        print(f"  optimal variables: {result.results.variables}")
        print(f"  optimal objective: {result.results.target_objective}")
        assert np.allclose(result.results.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
