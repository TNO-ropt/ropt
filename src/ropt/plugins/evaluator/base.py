"""Defines base classes for optimization workflow plugins and components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult


class EvaluatorPlugin(Plugin):
    """Abstract base class for evaluator plugins.

    This class defines the interface for plugins responsible for creating
    [`Evaluator`][ropt.plugins.evaluator.base.Evaluator] instances within
    an optimization workflow.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> Evaluator:
        """Create an Evaluator instance.

        This abstract class method serves as a factory for creating concrete
        [`Evaluator`][ropt.plugins.evaluator.base.Evaluator] objects. Plugin
        implementations must override this method to return an instance of their
        specific `Evaluator` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        an evaluator provided by this plugin is requested.

        The `name` argument specifies the requested evaluator, potentially
        in the format `"plugin-name/method-name"` or just `"method-name"`.
        Implementations can use this `name` to vary the created evaluator if
        the plugin supports multiple evaluator types.

        Args:
            name:    The requested evaluator name (potentially plugin-specific).
            kwargs:  Additional arguments for custom configuration.

        Returns:
            An initialized instance of an `Evaluator` subclass.
        """


class Evaluator(ABC):
    """Abstract base class for evaluator components within an optimization workflow.

    Subclasses must implement the abstract
    [`eval`][ropt.plugins.evaluator.base.Evaluator.eval] method, which is responsible
    for performing the actual evaluation of variables using an
    [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] and returning an
    [`EvaluatorResult`][ropt.evaluator.EvaluatorResult].

    `Evaluator` instances are typically created using the
    [`create_evaluator`][ropt.workflow.create_evaluator] function.
    """

    @abstractmethod
    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate objective and constraint functions for given variables.

        This method defines function evaluator callback, which calculates
        objective and constraint functions for a set of variable vectors,
        potentially for a subset of realizations and perturbations.

        Args:
            variables: The matrix of variables to evaluate. Each row represents
                       a variable vector.
            context:   The evaluation context, providing additional information
                       about the evaluation.

        Returns:
            An evaluation results object containing the calculated values.

        Tip: Reusing Objectives and Constraints
            When defining multiple objectives, there may be a need to reuse the
            same objective or constraint value multiple times. For instance, a
            total objective could consist of the mean of the objectives for each
            realization, plus the standard deviation of the same values. This
            can be implemented by defining two objectives: the first calculated
            as the mean of the realizations, and the second using a function
            estimator to compute the standard deviations. The optimizer is
            unaware that both objectives use the same set of realizations. To
            prevent redundant calculations, the evaluator should compute the
            results of the realizations once and return them for both
            objectives.
        """
