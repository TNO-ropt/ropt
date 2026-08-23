"""A custom realization filter with the high-level ``ropt.simple`` API.

A realization filter reweights the realizations of an ensemble at each
evaluation, letting the optimizer target a robust statistic instead of the mean.
This example implements ``MedianFilter``, which puts all weight on the
realization with the median objective. Since the filter is passed as an
instance, it needs no plugin registration: it is added to the top-level
``realization_filters`` list and referenced by index from the ``objectives``
section.
"""

from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.config import RealizationFilterConfig
from ropt.context import EnOptContext
from ropt.realization_filter import RealizationFilter
from ropt.simple import EvaluateResult, EvaluationFunctionContext, optimize

DIM = 5
UNCERTAINTY = 0.1
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


class MedianFilter(RealizationFilter):
    """Assign all weight to the realization with the median objective."""

    def __init__(
        self,
        filter_config: RealizationFilterConfig,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Create the filter.

        Args:
            filter_config: The filter configuration, unused by this filter.
        """
        self._realization_count: int

    def init(self, context: EnOptContext) -> None:
        """Store the number of realizations.

        Args:
            context: The optimization context.
        """
        self._realization_count = len(context.realizations.weights)

    def get_realization_weights(
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,  # ruff: ignore[unused-method-argument]
    ) -> NDArray[np.float64]:
        """Give the realization with the median objective a weight of one.

        Args:
            objectives:  The objective values for each realization.
            constraints: The constraint values, unused by this filter.

        Returns:
            The weights for each realization, zero for all but the median.
        """
        # The objectives have one row per realization and one column per
        # objective, hence the indexing to get the only objective there is. A
        # real filter would also handle failed realizations, which carry nan
        # values, for instance by giving them a weight of zero.
        order = np.argsort(objectives[:, 0])
        weights = np.zeros(self._realization_count, dtype=np.float64)
        weights[order[self._realization_count // 2]] = 1.0
        return weights


def report(result: EvaluateResult) -> None:
    """Print the objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


def main() -> None:
    """Run the median-filtered ensemble optimization and check the result."""
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
        # the nonlinear constraints that use them.
        "realization_filters": [
            MedianFilter(RealizationFilterConfig(method="median")),
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
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
