"""Spread ``optimize_many``'s arguments over its runs.

Each argument is either a single value shared by every run, or a sequence with
one entry per run. The sequences set the number of runs and must agree; single
values are repeated to match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ._function import EvaluationFunction
    from ._report import ReportCallback


def broadcast_runs(
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
) -> list[tuple[dict[str, Any], ArrayLike, EvaluationFunction]]:
    """Pair up the configuration, start point and function of each run.

    Args:
        config:   The configuration, or one per run.
        x0:       The initial variable vector, or one per row.
        function: The evaluation function, or one per run.

    Returns:
        One `(config, x0, function)` triple per run.

    Raises:
        ValueError: If `x0` is not a vector or a matrix, or if the sequences
                    given disagree on the number of runs.
    """
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
    """Give each run its report callback.

    Args:
        report: A callback shared by every run, one per run, or `None`.
        count:  The number of runs.

    Returns:
        One callback, or `None`, per run.
    """
    if report is None:
        return [None] * count
    if callable(report):
        return [report] * count
    return _sized(list(report), count, "report")


def broadcast_metadata(
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None, count: int
) -> list[dict[str, Any] | None]:
    """Give each run its metadata dictionary.

    Args:
        metadata: A dictionary shared by every run, one per run, or `None`.
        count:    The number of runs.

    Returns:
        One dictionary, or `None`, per run.
    """
    if metadata is None:
        return [None] * count
    if isinstance(metadata, Mapping):
        return [metadata] * count
    return _sized(list(metadata), count, "metadata")


def _sized(values: list[Any], count: int, name: str) -> list[Any]:
    if len(values) != count:
        msg = f"{name} sequence length must match the number of runs."
        raise ValueError(msg)
    return values
