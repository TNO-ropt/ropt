"""A custom function estimator with the high-level ``ropt.simple`` API.

A function estimator reduces the objective values of all realizations to the
single value the optimizer works with, together with the matching gradient. This
example implements ``GeometricMean``, which replaces the built-in weighted mean
with a weighted geometric mean.

The geometric mean averages relative rather than absolute differences: it is the
value that minimizes the squared differences of the logarithms, where the
arithmetic mean minimizes the squared differences of the values themselves. It
is defined for positive numbers only, which is why this example is tied to the
Rosenbrock function, a sum of squares that is never negative. See
[Geometric mean](https://en.wikipedia.org/wiki/Geometric_mean).

The estimator is **registered** with ``register_plugin``, which makes it
available exactly like an installed one: it is selected from the configuration
by its ``"plugin/method"`` string. An estimator defined in a script or a
notebook cannot be found through an entry point, and registering is what closes
that gap.
"""

from typing import Any, ClassVar

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.config import FunctionEstimatorConfig
from ropt.function_estimator import FunctionEstimator
from ropt.plugins import MethodSpec, register_plugin
from ropt.results import FunctionResults
from ropt.simple import EvaluationFunctionContext, optimize

DIM = 5
UNCERTAINTY = 0.1
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


class GeometricMean(FunctionEstimator):
    """Reduce the realizations to their weighted geometric mean.

    The mean is evaluated as ``exp(sum(weights * log(functions)))``, the
    logarithmic form that avoids the overflow of a long product. See
    [Geometric mean](https://en.wikipedia.org/wiki/Geometric_mean).
    """

    # The methods this plugin provides, as an installed plugin declares them.
    methods: ClassVar[MethodSpec] = {"geometric"}

    def __init__(self, estimator_config: FunctionEstimatorConfig) -> None:
        """Create the estimator.

        Args:
            estimator_config: The estimator configuration, unused by this estimator.
        """

    def init(self, *, merge_realizations: bool) -> None:  # ruff: ignore[no-self-use]
        """Refuse gradients that arrive already merged.

        Args:
            merge_realizations: Whether the gradients arrive merged.

        Raises:
            ValueError: If the gradients arrive merged.
        """
        # Merged gradients no longer carry the per-realization values the chain
        # rule in `calculate_gradient` divides by.
        if merge_realizations:
            msg = "The geometric mean does not support merged gradients."
            raise ValueError(msg)

    def calculate_function(  # ruff: ignore[no-self-use]
        self,
        functions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate the realization values into their geometric mean.

        Args:
            functions: The objective value of each realization, all positive.
            weights:   The weight of each realization, summing to one.

        Returns:
            The weighted geometric mean of the values.
        """
        return np.exp(weights @ np.log(functions))

    def calculate_gradient(  # ruff: ignore[no-self-use]
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate the realization gradients with the chain rule.

        Differentiating ``exp(sum(weights * log(functions)))`` scales each
        realization gradient by ``weights / functions`` and multiplies the sum
        by the geometric mean itself.

        Args:
            functions: The objective value of each realization, all positive.
            gradient:  The gradient of each realization, one column per realization.
            weights:   The weight of each realization, summing to one.

        Returns:
            The gradient of the weighted geometric mean.
        """
        geometric_mean = np.exp(weights @ np.log(functions))
        return geometric_mean * (gradient @ (weights / functions))


def report(result: FunctionResults) -> None:
    """Print the objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


def main() -> None:
    """Run the geometric-mean ensemble optimization and check the result."""
    register_plugin("function_estimator", "custom", GeometricMean)

    realizations = 10
    config: dict[str, Any] = {
        "variables": {
            "variable_count": DIM,
            "perturbation_magnitudes": 1e-6,
        },
        "realizations": {
            "weights": [1.0] * realizations,
        },
        # Estimators are listed here and referred to by index by the objectives
        # or the nonlinear constraints that use them. Passing the instance
        # itself, `GeometricMean(FunctionEstimatorConfig(method="geometric"))`,
        # works too and needs no registration.
        "function_estimators": [
            {"method": "custom/geometric"},
        ],
        "objectives": {
            "function_estimators": [0],
        },
    }

    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)

    def rosenbrock(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        r = context.realization
        objective = 0.0
        for d_idx in range(DIM - 1):
            x, y = variables[d_idx : d_idx + 2]
            objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
        return float(objective)

    result = optimize(config, INITIAL_VALUES, rosenbrock, report=report)
    assert result.results is not None
    print(f"optimal variables: {result.results.variables}")
    print(f"optimal objective: {result.results.target_objective}")
    assert np.allclose(result.results.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
