"""This module implements the default cached evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ropt.plugins.plan.base import Evaluator, EventHandler, PlanComponent
from ropt.results import FunctionResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorContext, EvaluatorResult
    from ropt.plan import Plan


class DefaultCachedEvaluator(Evaluator):
    """An evaluator that caches results to avoid redundant computations.

    This evaluator attempts to retrieve previously computed function results
    from a cache before delegating to another evaluator. The cache is populated
    from `FunctionResults` objects stored by `EventHandler` instances specified
    as `sources`.

    When an evaluation is requested, for each variable vector and its
    corresponding realization, this evaluator searches through the `results`
    attribute of its `sources`. If a `FunctionResults` object is found where the
    `variables` match the input (within a small tolerance) and the `realization`
    also matches, the cached `objectives` and `constraints` from that
    `FunctionResults` object are used.

    If some, but not all, requested evaluations are found in the cache, this
    evaluator will mark the cached ones as inactive for the next evaluator in
    the chain and then call that evaluator to compute only the missing results.
    The final combined results (cached and newly computed) are then returned.

    This is particularly useful in scenarios where the same variable sets might
    be evaluated multiple times, for example, in iterative optimization
    algorithms or when restarting optimizations.
    """

    _EPS: Final[float] = float(np.finfo(np.float64).eps)

    def __init__(
        self,
        plan: Plan,
        tags: set[str] | None = None,
        clients: set[PlanComponent | str] | None = None,
        *,
        sources: Sequence[EventHandler] | None = None,
    ) -> None:
        """Initialize the DefaultCachedEvaluator.

        The `sources` argument should be a sequence of `EventHandler` instances.
        These handlers are expected to store `FunctionResults` in their
        `["results"]` attribute.

        Args:
            plan:    The parent plan instance.
            tags:    Optional tags for this evaluator.
            clients: Plan components (steps or tags) this evaluator serves.
            sources: `EventHandler` instances for retrieving cached results.
        """
        super().__init__(plan, tags, clients)
        self._sources = [] if sources is None else sources

    def eval_cached(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> tuple[EvaluatorResult, list[int]]:
        """Evaluate objective and constraint functions, utilizing a cache.

        This method implements the core evaluation logic. It returns not only
        the evaluation results but also the indices of any variable vectors that
        were retrieved from the cache. The `eval` method in this class does not
        utilize these indices. However, derived classes can overload `eval` and
        use this information to add further details to the results, such as
        populating the `evaluation_info` attribute of an `EvaluatorResult`.

        Args:
            variables: Matrix of variables to evaluate (each row is a vector).
            context:   The evaluation context.

        Returns:
            An `EvaluatorResult` and a list of indices of cached variable vectors.
        """
        cached: dict[int, tuple[NDArray[np.float64], NDArray[np.float64] | None]] = {}

        for idx in range(variables.shape[0]):
            objectives, constraints = self._get_from_cache(
                variables[idx, :], context.realizations[idx]
            )
            if objectives is not None:
                cached[idx] = (objectives, constraints)

        if cached:
            cached_indices = list(cached.keys())
            active = np.ones(variables.shape[0], dtype=np.bool_)
            active[cached_indices] = False
            context.active = active
        else:
            cached_indices = []

        evaluator = self.plan.get_evaluator(self)
        results = evaluator.eval(variables, context)

        for idx, (objectives, constraints) in cached.items():
            results.objectives[idx, :] = objectives
            if results.constraints is not None and constraints is not None:
                results.constraints[idx, :] = constraints

        return results, cached_indices

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate objective and constraint functions, utilizing a cache.

        For each input variable vector and its realization, this method first
        attempts to find a matching result in the cache provided by its
        `sources`.

        If a result is found in the cache, it's used directly. If not, the
        evaluation is delegated to the next evaluator in the plan.
        The `context.active` array is updated to indicate to the subsequent
        evaluator which evaluations are still pending; evaluations found in
        the cache are marked as inactive.

        Args:
            variables: Matrix of variables to evaluate (each row is a vector).
            context:   The evaluation context.

        Returns:
            An `EvaluatorResult` with calculated or cached values.
        """
        result, _ = self.eval_cached(variables, context)
        return result

    def _get_from_cache(
        self, variables: NDArray[np.float64], realization: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        for results in _get_results(self._sources):
            if results.realizations.active_realizations[realization] and np.allclose(
                results.evaluations.variables, variables, rtol=0.0, atol=self._EPS
            ):
                objectives = results.evaluations.objectives[realization, :]
                if results.evaluations.constraints is None:
                    return objectives, None
                return objectives, results.evaluations.constraints[realization, :]
        return None, None


def _get_results(sources: Sequence[EventHandler]) -> Iterable[FunctionResults]:
    for source in sources:
        results = source["results"]
        if results is None:
            continue
        if isinstance(results, FunctionResults):
            yield results
        else:
            for item in results:
                if isinstance(item, FunctionResults):
                    yield item
