"""Nested optimization: an inner run per outer evaluation, on its own pool.

Each evaluation of the outer optimization runs an inner optimization over the
remaining variables. The two layers use **different** pools, which is what makes
this safe: the outer evaluations run on a thread pool, so each stays in this
process and can reach the inner pool, and the inner evaluations run on a process
pool of their own. Handing the inner run the pool it is already running on would
instead be refused, since it would wait for the workers it occupies.

The inner runs all feed one shared ``DataFrameHandler``. They overlap, so a
shared group is what makes that safe: the group serializes every run's results
through a single dispatcher. Each inner run tags its results with the outer
evaluation that started it, so every row in the frame can be traced back.
"""

from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.enums import VariableType
from ropt.simple import (
    DataFrameHandler,
    EvaluationFunction,
    EvaluationFunctionContext,
    SharedHandlers,
    WorkerPool,
    optimize,
    session,
)

DIM = 4
REALIZATIONS = 10
MASK = [True, True, False, False]
INNER_CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "mask": MASK,
        "lower_bounds": 0.0,
        "upper_bounds": 10.0,
    },
    "realizations": {"weights": [1.0] * REALIZATIONS},
    "optimizer": {"max_functions": 10},
}

OUTER_CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "mask": np.logical_not(MASK),
        "lower_bounds": 0.0,
        "upper_bounds": 10.0,
        "types": VariableType.INTEGER,
    },
    "realizations": {"weights": [1.0]},
    "backend": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "parallel": True,
        "max_iterations": 10,
    },
}
INITIAL_VALUES = [1.0, 1.0, 1.0, 1.0]
UNCERTAINTY = 0.1


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> float:
    """The Rosenbrock objective for one realization of the inner problem.

    Defined at module level, and closing over nothing, so it can be pickled
    into the inner process pool.

    Args:
        variables: The variable vector to evaluate.
        context:   The evaluation context, giving the realization index.
        a:         The per-realization ``a`` parameters.
        b:         The per-realization ``b`` parameters.

    Returns:
        The Rosenbrock objective at ``variables``.
    """
    objective = 0.0
    scaled = variables / np.arange(1, DIM + 1)
    for idx in range(DIM - 1):
        x, y = scaled[idx : idx + 2]
        r = context.realization
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return float(objective)


def inner_optimization(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    *,
    pool: WorkerPool,
    group: SharedHandlers,
    function: EvaluationFunction,
) -> float:
    """Evaluate one outer point by optimizing the inner variables at it.

    Runs in a thread of the outer pool, so the inner pool and the shared group
    are live objects here rather than copies.

    Args:
        variables: The outer variable vector to evaluate.
        context:   The evaluation context, identifying this outer evaluation.
        pool:      The pool the inner evaluations run on.
        group:     The shared handlers every inner run feeds.
        function:  The objective the inner optimization minimizes.

    Returns:
        The best inner objective found at this outer point.
    """
    result = optimize(
        INNER_CONFIG,
        np.where(MASK, INITIAL_VALUES, variables),
        function,
        pool=pool,
        handlers=[group],
        # Within a batch only (batch_id, eval_idx) is unique: several rows share
        # a realization, so realization alone would not identify the caller.
        metadata={"outer_batch": context.batch_id, "outer_eval": context.eval_idx},
    )
    assert result.target_objective is not None
    return result.target_objective


def main() -> None:
    """Run the outer optimization, collecting every inner result in one frame."""
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    # polars keeps the key columns as ordinary columns instead of an index, so
    # the frame can be filtered by outer evaluation directly.
    tables = DataFrameHandler(backend="polars")
    tables.add_table(
        "inner",
        "functions",
        {
            "metadata.outer_batch": "Outer-batch",
            "metadata.outer_eval": "Outer-eval",
            "batch_id": "Inner-batch",
            "functions.target_objective": "Objective",
            "evaluations.variables": "Variable",
        },
    )

    with session() as active:
        inner_pool = active.process_pool(workers=2, bundle_size=0)
        outer_pool = active.thread_pool(workers=2)
        group = active.shared_handlers(tables)
        optimize(
            OUTER_CONFIG,
            INITIAL_VALUES,
            partial(
                inner_optimization,
                pool=inner_pool,
                group=group,
                function=partial(rosenbrock, a=a, b=b),
            ),
            pool=outer_pool,
        )

    frame = tables["inner"]
    assert frame is not None
    print(frame)

    # The optimum has to be read from the shared frame rather than from the
    # outer result: the outer layer only ever sees its own variables, and holds
    # the inner ones at their initial values.
    best = frame.sort("Objective").row(0, named=True)
    variables = [best[f"Variable,{idx}"] for idx in range(DIM)]
    print(f"\nbest inner objective: {best['Objective']}")
    print(f"at variables:         {variables}")

    # Rosenbrock is scaled by 1..DIM, so its minimum sits at [1, 2, ..., DIM].
    assert np.allclose(variables, np.arange(1, DIM + 1), atol=1e-1)
    assert best["Objective"] < 1.0

    # Every inner result carries the outer evaluation that produced it, and the
    # inner runs share one pool, so their batch IDs never collide.
    assert frame.height > 0
    assert frame["Inner-batch"].n_unique() == frame.height
    assert frame["Outer-batch"].null_count() == 0


if __name__ == "__main__":
    main()
