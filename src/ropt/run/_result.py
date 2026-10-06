"""Result objects returned by a run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ropt.enums import ExitCode
    from ropt.results import FunctionResults, GradientResults


@dataclass
class OptimizationResult:
    """The outcome of a single optimization run.

    A run ends at one evaluation, the best one it found, and that evaluation is
    carried unchanged on `results`. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Attributes:
        exit_code: Why the optimization terminated.
        results:   The best evaluation, or `None` if there was no valid result.
        gradient:  The gradient at that evaluation, if one was computed there.
    """

    exit_code: ExitCode
    results: FunctionResults | None
    gradient: GradientResults | None = None


@dataclass
class EvaluationResult[T]:
    """The outcome of a single evaluation run.

    What `results` holds depends on which method produced it: one
    [`FunctionResults`][ropt.results.FunctionResults] from
    [`evaluate`][ropt.evaluate], or `None` if the evaluation was cut off;
    one per vector from [`evaluate_batch`][ropt.evaluate_batch], or an
    empty tuple. An evaluation is a single batch, so it produces either every
    result or none.

    Attributes:
        exit_code: Why the evaluation ended.
        results:   What the evaluation produced.
    """

    exit_code: ExitCode
    results: T
