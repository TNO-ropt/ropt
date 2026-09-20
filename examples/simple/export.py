"""Export results to a pandas or polars frame with the `ropt.simple` API.

A `Results` object holds everything one evaluated variable vector produced,
with each field indexed by its own axes. `to_pandas` and `to_polars` turn a
single result into a frame, while `results_to_pandas` and
`results_to_polars` turn a sequence of them into one aggregated frame.

Fields are named by dotted paths and given as an ordered sequence, so the
columns come out in the order asked for. Axes stay as row labels unless
`unstack` pivots them into columns; an aggregated frame pivots every axis
except `realization` and `perturbation`.

The example runs on polars by default; pass `-p`/`--pandas` to use pandas
instead. Only the library actually used needs to be installed.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.enums import AxisName
from ropt.results import FunctionResults, results_to_pandas, results_to_polars
from ropt.simple import (
    EvaluationFunctionResult,
    HistoryHandler,
    optimize,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ropt.simple import EvaluationFunctionContext

DIM = 3
SHIFTS = np.array([0.9, 1.1])  # one uncertain shift per realization
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "objectives": {"weights": [0.75, 0.25]},
    "realizations": {"weights": [1.0] * len(SHIFTS)},
    # Without these the axes are labelled with plain integers instead.
    "names": {
        "variable": ("x", "y", "z"),
        "objective": ("val", "cost"),
        "realization": ("r0", "r1"),
    },
}


def objective(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    """Two objectives for one realization of the problem.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The squared and the worst-case residual at `variables`.
    """
    residual = variables - SHIFTS[context.realization]
    return EvaluationFunctionResult(
        objectives=np.array([np.sum(residual**2), np.abs(residual).max()])
    )


def _export_pandas(results: Sequence[FunctionResults]) -> None:
    """Show the three shapes using pandas, which keeps the axes in an index.

    Args:
        results: The results collected by the run.
    """
    # Every axis of a selected field becomes a row label, under `batch_id`.
    # --8<-- [start:stacked]
    stacked = results[0].to_pandas(["evaluations.objectives"])
    # --8<-- [end:stacked]
    print(f"stacked\n{stacked}\n")
    assert list(stacked.index.names) == ["batch_id", "realization", "objective"]

    # `unstack` pivots an axis into columns, one per label on that axis.
    # --8<-- [start:unstacked]
    unstacked = results[0].to_pandas(
        ["evaluations.objectives"], unstack=[AxisName.OBJECTIVE]
    )
    # --8<-- [end:unstacked]
    print(f"unstacked\n{unstacked}\n")
    assert list(unstacked.columns.get_level_values(level=0)) == [
        ("evaluations.objectives", "val"),
        ("evaluations.objectives", "cost"),
    ]

    # An aggregated frame holds every result, and follows the field order given.
    aggregated = results_to_pandas(
        results, ["variables", "target_objective"], result_type="functions"
    )
    print(f"aggregated\n{aggregated}")
    assert list(aggregated.columns.get_level_values(level=0)) == [
        ("variables", "x"),
        ("variables", "y"),
        ("variables", "z"),
        "target_objective",
    ]
    assert len(aggregated) == len(results)


def _export_polars(results: Sequence[FunctionResults]) -> None:
    """Show the same three shapes using polars, which has no index.

    Args:
        results: The results collected by the run.
    """
    stacked = results[0].to_polars(["evaluations.objectives"])
    print(f"stacked\n{stacked}\n")
    assert stacked.columns == [
        "batch_id",
        "realization",
        "objective",
        "evaluations.objectives",
    ]

    unstacked = results[0].to_polars(
        ["evaluations.objectives"], unstack=[AxisName.OBJECTIVE]
    )
    print(f"unstacked\n{unstacked}\n")
    assert unstacked.columns == [
        "batch_id",
        "realization",
        "evaluations.objectives,val",
        "evaluations.objectives,cost",
    ]

    # --8<-- [start:aggregated]
    aggregated = results_to_polars(
        results, ["variables", "target_objective"], result_type="functions"
    )
    # --8<-- [end:aggregated]
    print(f"aggregated\n{aggregated}")
    assert aggregated.columns == [
        "batch_id",
        "variables,x",
        "variables,y",
        "variables,z",
        "target_objective",
    ]
    assert aggregated.height == len(results)


def main(*, pandas: bool = False) -> None:
    """Export one result, then every result, in a few different shapes.

    Args:
        pandas: Use pandas instead of polars.
    """
    history = HistoryHandler()
    optimize(CONFIG, np.zeros(DIM), objective, handlers=[history])
    results = [item for item in history.results if isinstance(item, FunctionResults)]
    assert results

    if pandas:
        _export_pandas(results)
    else:
        _export_polars(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pandas",
        action="store_true",
        help="Use pandas instead of polars.",
    )
    main(pandas=parser.parse_args().pandas)
