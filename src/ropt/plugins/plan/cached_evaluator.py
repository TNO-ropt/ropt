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

    def __init__(
        self,
        plan: Plan,
        tags: set[str] | None = None,
        clients: set[PlanComponent | str] | None = None,
        *,
        sources: list[EventHandler] | None = None,
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
        self._sources: list[EventHandler] = [] if sources is None else sources

    def eval_cached(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> tuple[EvaluatorResult, dict[int, tuple[int, FunctionResults]]]:
        """Evaluate objective and constraint functions, utilizing a cache.

        This method implements the core evaluation logic. It returns not only
        the evaluation results but also the function results that were retrieved
        from cache. The `eval` method in this class does not utilize these
        indices. However, derived classes can overload `eval` and use this
        information to add further details to the results, such as populating
        the `evaluation_info` attribute of an `EvaluatorResult`.

        The cache hits that are returned consists of a dictionary where the keys
        are the indices of the variable vectors that were found in the cache,
        and the values are tuples containing the realization index of the cached
        vectors and the corresponding `FunctionResults` object. This allows the
        caller to know which evaluations were retrieved from cache and which
        were computed anew. The cached evaluations can then be retrieved from
        the `FunctionResults` object, using the realization index.

        Note:
            If the configuration stored in the context contains realization
            names, these are used to match the realizations of the requested
            evaluations, with those in the cached results. This means that the
            results may originate from a different optimization run, as long as
            the realization names are still valid. However, in this case the
            results used for finding cached values must also store the
            realization names, otherwise the cached results will not be found.

            If the configuration does not contain realization names, the
            realization indices are used to match the realizations of the
            requested evaluations. In this case the indices of the realizations
            in the cached results must match those of the requested evaluations,
            i.e. they must have been specified in the same order in the
            respective configurations.

        Args:
            variables: Matrix of variables to evaluate (each row is a vector).
            context:   The evaluation context.

        Returns:
            An `EvaluatorResult` and the cache hits.
        """
        cached: dict[int, tuple[int, FunctionResults]] = {}

        realization_names = context.config.names.get("realization", None)

        for idx in range(variables.shape[0]):
            realization_index = context.realizations[idx]
            realization_name = (
                realization_names[realization_index]
                if realization_names is not None
                else None
            )
            results, cached_realization_index = _get_from_cache(
                self._sources,
                variables[idx, :],
                realization_index,
                realization_name,
            )
            if results is not None:
                cached[idx] = (cached_realization_index, results)

        if cached:
            context.active[list(cached.keys())] = False
        evaluator_result = self.plan.get_evaluator(self).eval(variables, context)

        for idx, (realization, item) in cached.items():
            objectives = item.evaluations.objectives
            constraints = item.evaluations.constraints
            evaluator_result.objectives[idx, :] = objectives[realization, :]
            if evaluator_result.constraints is not None:
                assert constraints is not None
                evaluator_result.constraints[idx, :] = constraints[realization, :]

        return evaluator_result, cached

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

    def add_sources(self, sources: EventHandler | Sequence[EventHandler]) -> None:
        """Add one or more `EventHandler` sources to the evaluator.

        This method allows adding additional sources of cached results to the
        evaluator. The sources are expected to be `EventHandler` instances that
        store `FunctionResults`.

        Args:
            sources: `EventHandler` instances to add as a source.
        """
        if isinstance(sources, EventHandler):
            sources = [sources]
        self._sources.extend(sources)

    def remove_sources(self, sources: EventHandler | Sequence[EventHandler]) -> None:
        """Remove one or more `EventHandler` sources from the evaluator.

        This method allows removing previously added sources of cached results
        from the evaluator.

        Args:
            sources: `EventHandler` instances to remove as a source.
        """
        if isinstance(sources, EventHandler):
            sources = [sources]
        self._sources.extend(sources)


_EPS: Final[float] = float(np.finfo(np.float64).eps)


def _get_from_cache(
    sources: list[EventHandler],
    variables: NDArray[np.float64],
    realization_index: int,
    realization_name: str | None,
) -> tuple[FunctionResults | None, int]:
    for results in _get_results(sources):
        if realization_name is not None:
            names: tuple[str | int, ...] = results.config.names.get("realization", ())
            realization_index = list(names).index(realization_name)
            if realization_index < 0:
                continue
        if results.realizations.active_realizations[realization_index] and np.allclose(
            results.evaluations.variables, variables, rtol=0.0, atol=_EPS
        ):
            return results, realization_index
    return None, -1


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
