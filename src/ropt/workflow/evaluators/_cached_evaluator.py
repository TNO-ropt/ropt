"""This module implements the default cached evaluator."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

import numpy as np

from ropt._logging import get_logger
from ropt.results import FunctionResults
from ropt.workflow.event_handlers import EventHandler

from .base import Evaluator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

    from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult

_logger = get_logger(__name__)


class CachedEvaluator(Evaluator):
    """An evaluator that caches results to avoid redundant computations.

    Wraps another evaluator, retrieving previously computed results from
    `EventHandler` sources before delegating uncached evaluations.

    See [Optimization Workflows](../usage/workflows.md#evaluators) for full
    details on cache matching, realization name handling, and source
    management.
    """

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        sources: Sequence[EventHandler] | set[EventHandler] | None = None,
        hits_key: str | None = None,
    ) -> None:
        """Initialize the CachedEvaluator.

        The `sources` argument should be a sequence of `EventHandler` instances.
        These handlers are expected to store `FunctionResults` in their
        `["results"]` attribute.

        Args:
            evaluator: The evaluator to cache.
            sources:   `EventHandler` instances for retrieving cached results.
            hits_key:   Optional key for storing cache-hits in `metadata`.
        """
        super().__init__()
        self._evaluator = evaluator
        self._sources: list[EventHandler] = [] if sources is None else list(sources)
        self._hits_key = hits_key

    def eval_cached(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> tuple[EvaluationBatchResult, dict[int, tuple[int, FunctionResults]]]:
        """Evaluate using cache, returning both results and cache-hit info.

        Returns the evaluation results together with a dictionary of cache
        hits. The dictionary keys are evaluation indices found in cache;
        values are tuples of (realization index, cached FunctionResults).

        Derived classes can override `eval` and call this method to access
        cache-hit information for populating `metadata`.

        Note:
            If realization names are configured, they are used for matching
            (allowing cache hits across runs). Otherwise realization indices
            are used.

        Args:
            variables:         Matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            An `EvaluationBatchResult` and the cache hits.
        """
        cached: dict[int, tuple[int, FunctionResults]] = {}

        sources = tuple(self._sources)
        names = evaluator_context.context.names.get("realization")
        for idx in range(variables.shape[0]):
            realization_index = evaluator_context.realizations[idx]
            realization_name = names[realization_index] if names is not None else None
            results, cached_realization_index = _get_from_cache(
                sources,
                variables[idx, :],
                realization_index,
                realization_name,
            )
            if results is not None:
                cached[idx] = (cached_realization_index, results)

        if cached:
            _logger.debug(
                "Cache: %d/%d evaluations served from cache",
                len(cached),
                variables.shape[0],
            )
            active = evaluator_context.active.copy()
            active[list(cached.keys())] = False
            evaluator_context = replace(evaluator_context, active=active)

        evaluator_result = self._evaluator.eval(variables, evaluator_context)

        if self._hits_key is not None:
            hits = np.zeros(variables.shape[0], dtype=np.bool_)
            hits[list(cached.keys())] = True
            evaluator_result.metadata[self._hits_key] = hits

        for idx, (realization, item) in cached.items():
            objectives = item.evaluations.objectives
            constraints = item.evaluations.constraints
            evaluator_result.objectives[idx, :] = objectives[realization, :]
            if evaluator_result.constraints is not None:
                assert constraints is not None
                evaluator_result.constraints[idx, :] = constraints[realization, :]

        return evaluator_result, cached

    def eval(
        self, variables: NDArray[np.float64], context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate using cache, delegating uncached evaluations.

        Args:
            variables: Matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            An `EvaluationBatchResult` with calculated or cached values.
        """
        result, _ = self.eval_cached(variables, context)
        return result

    def add_sources(self, sources: EventHandler | Sequence[EventHandler]) -> None:
        """Add one or more `EventHandler` sources.

        Args:
            sources: `EventHandler` instances to add as a source.
        """
        if isinstance(sources, EventHandler):
            sources = [sources]
        self._sources.extend(sources)


_EPS: Final[float] = float(np.finfo(np.float64).eps)


def _get_from_cache(
    sources: Sequence[EventHandler],
    variables: NDArray[np.float64],
    realization_index: int,
    realization_name: str | None,
) -> tuple[FunctionResults | None, int]:
    for results in _get_results(sources):
        if realization_name is not None:
            names: tuple[str | int, ...] = results.names.get("realization", ())
            realization_index = list(names).index(realization_name)
            if realization_index < 0:
                continue
        if results.realizations.evaluated_realizations[
            realization_index
        ] and np.allclose(
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
