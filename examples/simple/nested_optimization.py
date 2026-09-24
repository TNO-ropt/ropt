"""Nested optimization: an inner run per outer evaluation, on its own pool.

Each evaluation of the outer optimization runs an inner optimization over the
remaining variables. The two layers use **different** pools, which is what makes
this safe: the outer evaluations run on a thread pool, so each stays in this
process and can reach the inner pool, and the inner evaluations run on a process
pool of their own. Handing the inner run the pool it is already running on would
instead be refused, since it would wait for the workers it occupies.

The inner runs all feed one `DataFrameHandler`. They overlap, which the handler
allows: `handle_event` serializes its own calls, so a second run waits for the
first. Each inner run tags its results with the outer evaluation that started
it, so every row in the frame can be traced back.

The outer optimizer works on integer variables and revisits points it has
already tried. The outer evaluation function keeps a memo of the objectives it
has computed, so a repeat returns the stored value instead of running another
inner optimization.
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
    WorkerPool,
    optimize,
    session,
)

# --8<-- [start:configs]
DIM = 4
REALIZATIONS = 5
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
    "optimizer": {"max_functions": 8},
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
        "max_iterations": 4,
    },
}
# --8<-- [end:configs]
INITIAL_VALUES = [1.0, 1.0, 1.0, 1.0]
UNCERTAINTY = 0.01


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
        a:         The per-realization `a` parameters.
        b:         The per-realization `b` parameters.

    Returns:
        The Rosenbrock objective at `variables`.
    """
    objective = 0.0
    scaled = variables / np.arange(1, DIM + 1)
    for idx in range(DIM - 1):
        x, y = scaled[idx : idx + 2]
        r = context.realization
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return float(objective)


def inner_optimization(  # ruff: ignore[too-many-arguments]
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    *,
    pool: WorkerPool,
    tables: DataFrameHandler,
    function: EvaluationFunction,
    memo: dict[tuple[float, ...], float],
) -> float:
    """Evaluate one outer point by optimizing the inner variables at it.

    Runs in a thread of the outer pool, so the inner pool, the handler and the
    memo are live objects here rather than copies.

    Args:
        variables: The outer variable vector to evaluate.
        context:   The evaluation context, identifying this outer evaluation.
        pool:      The pool the inner evaluations run on.
        tables:    The handler every inner run feeds.
        function:  The objective the inner optimization minimizes.
        memo:      Objectives already computed, keyed by outer point.

    Returns:
        The best inner objective found at this outer point.
    """
    # --8<-- [start:inner]
    key = (context.realization, *variables.tolist())
    if key in memo:
        return memo[key]

    result = optimize(
        INNER_CONFIG,
        np.where(MASK, INITIAL_VALUES, variables),
        function,
        pool=pool,
        handlers=[tables],
        # A whole inner batch goes to one worker: the parallelism comes from the
        # outer runs.
        bundle_size=0,
        # Within a batch only (batch_id, eval_idx) is unique: several rows share
        # a realization, so realization alone would not identify the caller.
        metadata={"outer_batch": context.batch_id, "outer_eval": context.eval_idx},
    )
    assert result.results is not None
    assert result.results.target_objective is not None
    memo[key] = float(result.results.target_objective)
    return memo[key]
    # --8<-- [end:inner]


def main() -> None:
    """Run the outer optimization, collecting every inner result in one frame."""
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    # polars keeps the key columns as ordinary columns instead of an index, so
    # the frame can be filtered by outer evaluation directly.
    tables = DataFrameHandler()
    tables.add_table(
        "inner",
        "functions",
        {
            "metadata.outer_batch": "Outer-batch",
            "metadata.outer_eval": "Outer-eval",
            "batch_id": "Inner-batch",
            "target_objective": "Objective",
            "variables": "Variable",
        },
    )

    # --8<-- [start:run]
    # A plain dict reaches the outer evaluations because they run on threads, in
    # this process; on a process pool each worker would get an empty copy.
    memo: dict[tuple[float, ...], float] = {}
    with session() as active:
        inner_pool = active.process_pool(workers=2)
        outer_pool = active.thread_pool(workers=2)
        optimize(
            OUTER_CONFIG,
            INITIAL_VALUES,
            partial(
                inner_optimization,
                pool=inner_pool,
                tables=tables,
                function=partial(rosenbrock, a=a, b=b),
                memo=memo,
            ),
            pool=outer_pool,
        )
    # --8<-- [end:run]

    frame = tables["inner"]
    assert frame is not None
    print(frame)

    # The optimum has to be read from the shared frame rather than from the
    # outer result: the outer layer only ever sees its own variables, and holds
    # the inner ones at their initial values.
    # --8<-- [start:best]
    best = frame.sort("Objective").row(0, named=True)
    variables = [best[f"Variable,{idx}"] for idx in range(DIM)]
    # --8<-- [end:best]
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
