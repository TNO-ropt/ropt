"""Scaling, and the two domains a result is reported in.

`ropt` does not hand your numbers to the optimizer unchanged. Variables are
divided by their `scales`, and objectives can be scaled too: setting
`auto_scale` divides each objective by its own value in the first batch, which
brings an objective of any magnitude close to one.

A result therefore carries the same quantity twice. `variables` and
`functions.objectives` are in the units configured here, while
`scaled.variables` and `scaled.functions.objectives` are in the units the
optimizer works in. The `target_objective` that a run reports is the
optimizer's number, and has no counterpart in the configured units.

The distinction matters when comparing runs. Switching `auto_scale` on makes
the reported objective much smaller while the solution stays where it was, so
two runs are only comparable through `functions.objectives` or the variables
themselves.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.results import FunctionResults
from ropt.simple import EvaluationFunctionContext, optimize

DIM = 5
UNCERTAINTY = 0.1
REALIZATIONS = 10
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
VARIABLE_SCALE = 10.0
# The objective starts far above one, so auto_scale divides it by a large factor.
MIN_AUTO_SCALE_FACTOR = 10.0


def run(
    label: str,
    config: dict[str, Any],
    objective: Callable[[NDArray[np.float64], EvaluationFunctionContext], float],
) -> FunctionResults:
    """Optimize, and report the optimum in both domains.

    Args:
        label:     The name to print for this run.
        config:    The configuration to run.
        objective: The evaluation function.

    Returns:
        The full result at the optimum.
    """
    result = optimize(config, INITIAL_VALUES, objective)
    # The summary carries the full result of the optimum, which is where both
    # domains live; the summary itself reports only the optimizer's.
    best = result.results
    assert best is not None
    assert best.functions is not None
    assert best.scaled.functions is not None
    assert best.target_objective is not None
    print(f"\n{label}")
    print(f"  variables          {np.round(best.variables, 4)}")
    print(f"  scaled.variables   {np.round(best.scaled.variables, 4)}")
    print(f"  objective          {best.functions.objectives[0]:.6f}   (your units)")
    print(
        f"  scaled objective   {best.scaled.functions.objectives[0]:.6f}   (optimizer)"
    )
    print(f"  target_objective   {float(best.target_objective):.6f}   (reported)")
    return best


def main() -> None:
    """Compare the two domains under objective and variable scaling."""
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

    plain: dict[str, Any] = {
        "variables": {
            "variable_count": DIM,
            "perturbation_magnitudes": 1e-6,
        },
        "realizations": {
            "weights": [1.0] * REALIZATIONS,
        },
    }
    plain_best = run("no scaling: the two domains agree", plain, rosenbrock)

    scaled_objective = {
        **plain,
        "objectives": {"auto_scale": True},
    }
    auto_best = run(
        "auto_scale: the reported objective shrinks", scaled_objective, rosenbrock
    )

    scaled_variables = {
        **plain,
        "variables": {**plain["variables"], "scales": [VARIABLE_SCALE] * DIM},
    }
    scaled_best = run(
        f"variable scales of {VARIABLE_SCALE}: the domains differ",
        scaled_variables,
        rosenbrock,
    )

    # Without scaling there is nothing to tell the two domains apart.
    assert plain_best.functions is not None
    assert plain_best.scaled.functions is not None
    assert np.allclose(plain_best.variables, plain_best.scaled.variables)
    assert np.allclose(
        plain_best.functions.objectives, plain_best.scaled.functions.objectives
    )

    # auto_scale divides the objective by its value in the first batch, so the
    # reported number drops while the solution does not move.
    assert auto_best.functions is not None
    assert auto_best.scaled.functions is not None
    factor = (
        auto_best.functions.objectives[0] / auto_best.scaled.functions.objectives[0]
    )
    print(f"\nauto_scale divided the reported objective by {factor:.1f}")
    assert factor > MIN_AUTO_SCALE_FACTOR
    assert np.allclose(auto_best.variables, plain_best.variables, atol=1e-2)

    # A variable scale is an exact change of units, both ways.
    assert np.allclose(
        scaled_best.variables, scaled_best.scaled.variables * VARIABLE_SCALE
    )

    for best in (plain_best, auto_best, scaled_best):
        assert np.allclose(best.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
