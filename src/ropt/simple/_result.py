"""Result objects returned by the high-level API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ropt.enums import ExitCode
    from ropt.results import FunctionResults


@dataclass
class OptimizationResult:
    """The outcome of a single optimization run.

    A run ends at one evaluation, the best one it found, and that evaluation is
    carried unchanged on `results`. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Attributes:
        exit_code: The exit code describing how the optimization terminated.
        results:   The best evaluation, or `None` if there was no valid result.
    """

    exit_code: ExitCode
    results: FunctionResults | None
