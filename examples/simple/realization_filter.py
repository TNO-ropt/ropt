"""A custom realization filter with the high-level ``ropt.simple`` API.

A realization filter reweights the realizations of an ensemble at each
evaluation, letting the optimizer target a robust statistic instead of the mean.
This example implements ``MedianFilter``, which puts all weight on the
realization with the median objective.

The filter is **registered** with ``register_plugin``, which makes it available
exactly like an installed one: it is selected from the configuration by its
``"plugin/method"`` string. A filter defined in a script or a notebook cannot be
found through an entry point, and registering is what closes that gap.
"""

from typing import Any, ClassVar

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.config import RealizationFilterConfig
from ropt.plugins import MethodSpec, register_plugin
from ropt.realization_filter import RealizationFilter
from ropt.results import FunctionResults
from ropt.simple import EvaluationFunctionContext, optimize

DIM = 5
UNCERTAINTY = 0.1
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


class MedianFilter(RealizationFilter):
    """Assign all weight to the realization with the median objective."""

    # The methods this plugin provides, as an installed plugin declares them.
    methods: ClassVar[MethodSpec] = {"median"}

    def __init__(
        self,
        filter_config: RealizationFilterConfig,
    ) -> None:
        """Create the filter.

        Args:
            filter_config: The filter configuration, unused by this filter.
        """

    def get_realization_weights(  # ruff: ignore[no-self-use]
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,  # ruff: ignore[unused-method-argument]
        *,
        objective_scales: NDArray[np.float64],  # ruff: ignore[unused-method-argument]
        maximize: NDArray[np.bool_],  # ruff: ignore[unused-method-argument]
        objective_weights: NDArray[np.float64],  # ruff: ignore[unused-method-argument]
    ) -> NDArray[np.float64]:
        """Give the realization with the median objective a weight of one.

        Args:
            objectives:        The objective values for each realization.
            constraints:       The constraint values, unused by this filter.
            objective_scales:  The objective scales, unused by this filter.
            maximize:          The objective directions, unused by this filter.
            objective_weights: The objective weights, unused by this filter.

        Returns:
            The weights for each realization, zero for all but the median.
        """
        # The objectives have one row per realization and one column per
        # objective, hence the indexing to get the only objective there is. A
        # real filter would also handle failed realizations, which carry nan
        # values, for instance by giving them a weight of zero.
        realization_count = objectives.shape[0]
        order = np.argsort(objectives[:, 0])
        weights = np.zeros(realization_count, dtype=np.float64)
        weights[order[realization_count // 2]] = 1.0
        return weights


def report(result: FunctionResults) -> None:
    """Print the objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


def main() -> None:
    """Run the median-filtered ensemble optimization and check the result."""
    register_plugin("realization_filter", "custom", MedianFilter)

    realizations = 10
    config: dict[str, Any] = {
        "variables": {
            "variable_count": DIM,
            "perturbation_magnitudes": 1e-6,
        },
        "realizations": {
            "weights": [1.0] * realizations,
        },
        # Filters are listed here and referred to by index by the objectives or
        # the nonlinear constraints that use them. Passing the instance itself,
        # `MedianFilter(RealizationFilterConfig(method="median"))`, works too
        # and needs no registration.
        "realization_filters": [
            {"method": "custom/median"},
        ],
        "objectives": {
            "realization_filters": [0],
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
