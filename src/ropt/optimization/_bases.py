"""This module defines the abstract base classes for optimization steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.results import FunctionResults, Results


class LabelStep(ABC):
    """A base for label steps."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Get the label."""


class BasicStep(ABC):
    """An abstract base for basic steps."""

    @abstractmethod
    def run(self) -> bool:
        """Run the step.

        Returns:
            True if a user abort occurred.
        """


class TrackerStep(ABC):
    """Abstract base class for trackers."""

    @property
    @abstractmethod
    def id(self) -> Optional[str]:
        """The ID of the tracker.

        Returns:
            The tracker ID.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the results."""

    @abstractmethod
    def track_results(self, results: Tuple[Results, ...], name: str) -> None:
        """Track results.

        Args:
            results: The results to track
            name:    Name of the step producing the results
        """


class OptimizerStep(ABC):
    """Abstract base for for optimization steps."""

    @abstractmethod
    def run(self, variables: Optional[NDArray[np.float64]]) -> bool:
        """Run the step.

        Args:
            variables: Optional variables to start running with

        Returns:
            True if a user abort occurred.
        """

    @abstractmethod
    def run_nested_plan(
        self, variables: NDArray[np.float64]
    ) -> Tuple[Optional[FunctionResults], bool]:
        """Run a nested plan from an optimization run.

        Args:
            variables: The variables to run with

        Returns:
            The result and whether the run was aborted.
        """

    @abstractmethod
    def start_evaluation(self) -> None:
        """Called before the optimizer starts an evaluation."""

    @abstractmethod
    def finish_evaluation(self, results: Tuple[Results, ...]) -> None:
        """Called after the optimizer finishes an evaluation.

        Args:
            results: The results produced by the evaluation.
        """


class EvaluatorStep(ABC):
    """Abstract base for for evaluation steps."""

    @abstractmethod
    def run(self, variables: Optional[NDArray[np.float64]]) -> bool:
        """Run the step.

        Args:
            variables: Optional variables to start running with

        Returns:
            True if a user abort occurred.
        """

    @abstractmethod
    def process(self, results: FunctionResults) -> None:
        """Process the results of the evaluation.

        Args:
            results: The results to process
        """
