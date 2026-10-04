"""Reuse the results at the starting point in a second run.

The first run records the function and gradient results at its initial point. A
second run starting from the same point is given those results, so the
evaluations there are not repeated. Its realization weights differ, so the same
recorded values are aggregated into a different objective.
"""

from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.results import FunctionResults, GradientResults
from ropt.simple import EvaluationFunctionContext, HistoryHandler, optimize

DIM = 3
REALIZATIONS = 3
PERTURBATIONS = 4
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-3,
    },
    "realizations": {"weights": [1.0] * REALIZATIONS},
    "gradient": {"number_of_perturbations": PERTURBATIONS},
    "optimizer": {"max_functions": 5},
}
INITIAL_VALUES = np.zeros(DIM)


class CountingObjective:
    """An ensemble objective that counts the evaluations it performs."""

    def __init__(self) -> None:
        """Start with an empty count."""
        self.count = 0

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        """Evaluate one realization and count the call.

        Args:
            variables: The variable vector to evaluate.
            context:   The context of this evaluation.

        Returns:
            The squared distance to this realization's target.
        """
        self.count += 1
        target = 0.5 + 0.1 * context.realization
        return float(((variables - target) ** 2).sum())


def main() -> None:
    """Record the results at the starting point, then reuse them."""
    history = HistoryHandler()
    first = CountingObjective()
    optimize(CONFIG, INITIAL_VALUES, first, handlers=[history])
    f0 = next(item for item in history.results if isinstance(item, FunctionResults))
    g0 = next(item for item in history.results if isinstance(item, GradientResults))

    config = deepcopy(CONFIG)
    config["realizations"]["weights"] = [3.0, 1.0, 1.0]
    reused = HistoryHandler()
    second = CountingObjective()
    optimize(config, INITIAL_VALUES, second, handlers=[reused], f0=f0, g0=g0)
    restarted = next(
        item for item in reused.results if isinstance(item, FunctionResults)
    )

    assert f0.functions is not None
    assert restarted.functions is not None
    print(f"evaluations in the first run:  {first.count}")
    print(f"evaluations in the second run: {second.count}")
    print(f"not repeated at the starting point: {REALIZATIONS * (1 + PERTURBATIONS)}")
    print(f"objective at the starting point, first run:  {f0.functions.objectives}")
    print(
        f"objective at the starting point, second run: {restarted.functions.objectives}"
    )

    # The same raw values, aggregated under the new weights.
    assert np.allclose(restarted.evaluations.objectives, f0.evaluations.objectives)
    assert not np.allclose(restarted.functions.objectives, f0.functions.objectives)


if __name__ == "__main__":
    main()
