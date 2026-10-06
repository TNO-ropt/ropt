"""Spread `optimize_many`'s arguments over its runs.

Each argument is either a single value shared by every run, or a sequence with
one entry per run. The sequences set the number of runs and must agree; single
values are repeated to match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.results import Results

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ropt.results import FunctionResults, GradientResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback


def broadcast_runs(
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
) -> list[tuple[dict[str, Any], ArrayLike, EvaluationFunction]]:
    configs = [config] if isinstance(config, Mapping) else list(config)
    functions = [function] if callable(function) else list(function)

    x0_array = np.asarray(x0, dtype=np.float64)
    if x0_array.ndim == 1:
        x0s: list[ArrayLike] = [x0_array]
    elif x0_array.ndim == 2:  # ruff: ignore[magic-value-comparison]
        x0s = list(x0_array)
    else:
        msg = "x0 must be a vector or a 2-D matrix of vectors."
        raise ValueError(msg)

    counts = {len(seq) for seq in (configs, functions, x0s) if len(seq) != 1}
    if len(counts) > 1:
        msg = "config, x0 and function sequences must have the same length."
        raise ValueError(msg)
    count = counts.pop() if counts else 1

    def _repeat(seq: list[Any]) -> list[Any]:
        return seq * count if len(seq) == 1 else seq

    return list(zip(_repeat(configs), _repeat(x0s), _repeat(functions), strict=True))


def broadcast_reports(
    report: ReportCallback | Sequence[ReportCallback] | None, count: int
) -> list[ReportCallback | None]:
    if report is None:
        return [None] * count
    if callable(report):
        return [report] * count
    return _sized(list(report), count, "report")


def broadcast_metadata(
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None, count: int
) -> list[dict[str, Any] | None]:
    if metadata is None:
        return [None] * count
    if isinstance(metadata, Mapping):
        return [metadata] * count
    return _sized(list(metadata), count, "metadata")


def broadcast_bundle_sizes(
    bundle_size: int | Sequence[int | None] | None, count: int
) -> list[int | None]:
    if bundle_size is None or isinstance(bundle_size, int):
        return [bundle_size] * count
    return _sized(list(bundle_size), count, "bundle_size")


def broadcast_initial_values[ResultT: Results](
    values: ResultT | Sequence[ResultT | None] | None, count: int, name: str
) -> list[ResultT | None]:
    if values is None or isinstance(values, Results):
        return [values] * count
    return _sized(list(values), count, name)


type RunArguments = tuple[
    list[tuple[dict[str, Any], ArrayLike, EvaluationFunction]],
    list[ReportCallback | None],
    list[dict[str, Any] | None],
    list[int | None],
    list[FunctionResults | None],
    list[GradientResults | None],
]


def broadcast_arguments(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
    *,
    report: ReportCallback | Sequence[ReportCallback] | None,
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None,
    bundle_size: int | Sequence[int | None] | None,
    f0: FunctionResults | Sequence[FunctionResults | None] | None,
    g0: GradientResults | Sequence[GradientResults | None] | None,
) -> RunArguments:
    runs = broadcast_runs(config, x0, function)
    count = len(runs)
    return (
        runs,
        broadcast_reports(report, count),
        broadcast_metadata(metadata, count),
        broadcast_bundle_sizes(bundle_size, count),
        broadcast_initial_values(f0, count, "f0"),
        broadcast_initial_values(g0, count, "g0"),
    )


def _sized(values: list[Any], count: int, name: str) -> list[Any]:
    if len(values) != count:
        msg = f"{name} sequence length must match the number of runs."
        raise ValueError(msg)
    return values
