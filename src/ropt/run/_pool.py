"""The worker pool: the workers that a run's evaluations are given to.

A pool comes from a session factory (`thread_pool`, `process_pool`,
`local_pool`, `hpc_pool`), which is what ties its lifetime to a session:
closing the session releases the pools it built.

A run is started on the pool it should evaluate on, so where its evaluations
happen is stated at the call site, and the session it belongs to follows from
the pool rather than from where the call was made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from ropt.exceptions import WorkflowError

from ._evaluate import _evaluate, _evaluate_batch
from ._offload import _offload
from ._optimize import _optimize, _optimize_many

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.components.executors import Executor
    from ropt.results import FunctionResults, GradientResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback
    from ._result import EvaluationResult, OptimizationResult
    from ._session import Session

_RELEASED = (
    "This pool was released when its session closed; build a new one inside an "
    "open session, for example `with session() as s: pool = s.thread_pool()`."
)


class WorkerPool:
    """The workers that a run's evaluations are given to.

    Built by a session factory such as
    [`thread_pool`][ropt.Session.thread_pool], and released when that
    session closes. Starting a run on a pool sets both where the
    evaluations happen and which session the run belongs to. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    This is the type to annotate with when your own code takes a pool.
    """

    def __init__(self, session: Session, executor: Executor) -> None:
        """Initialize a pool owned by a session.

        Args:
            session:  The session that built this pool.
            executor: The executor its evaluations run on.
        """
        self._session = session
        self._executor: Executor | None = executor

    @property
    def _live_executor(self) -> Executor:
        if self._executor is None:
            raise WorkflowError(_RELEASED)
        return self._executor

    def _release(self) -> None:
        # Dropping the executor is what releases its workers. The pool itself
        # may outlive this, as a name the caller still holds.
        self._executor = None

    def optimize(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any],
        x0: ArrayLike,
        function: EvaluationFunction,
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | None = None,
        constraint_tolerance: float = 1e-10,
        bundle_size: int | None = None,
        metadata: dict[str, Any] | None = None,
        f0: FunctionResults | None = None,
        g0: GradientResults | None = None,
        report_gradients: bool = False,
    ) -> OptimizationResult:
        """Run a single optimization, evaluating on this pool.

        See [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            config:               The optimization configuration.
            x0:                   The initial variable vector.
            function:             The per-realization evaluation function.
            handlers:             Optional handlers, called in the order listed.
            report:               Optional callback invoked per evaluation.
            constraint_tolerance: The tolerance within which a constraint holds.
            bundle_size:          Evaluations per worker task, `None` for the
                                  pool's own.
            metadata:             Optional dictionary attached to every result.
            f0:                   Optional function results at `x0`.
            g0:                   Optional gradient results at `x0`.
            report_gradients:     Whether `report` also receives gradient results.

        Returns:
            An [`OptimizationResult`][ropt.OptimizationResult].

        Raises:
            WorkflowError: If this pool's session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _optimize(
            self._session,
            self._live_executor,
            config,
            x0,
            function,
            handlers=handlers,
            report=report,
            constraint_tolerance=constraint_tolerance,
            bundle_size=bundle_size,
            metadata=metadata,
            optimize_many_signal=None,
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )

    def optimize_many(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any] | Sequence[dict[str, Any]],
        x0: ArrayLike,
        function: EvaluationFunction | Sequence[EvaluationFunction],
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | Sequence[ReportCallback] | None = None,
        limit: int | None = None,
        constraint_tolerance: float = 1e-10,
        bundle_size: int | Sequence[int | None] | None = None,
        metadata: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        f0: FunctionResults | Sequence[FunctionResults | None] | None = None,
        g0: GradientResults | Sequence[GradientResults | None] | None = None,
        report_gradients: bool = False,
    ) -> tuple[OptimizationResult, ...]:
        """Run several optimizations concurrently, all evaluating on this pool.

        Each of `config`, `x0` and `function` is either one value used by every
        run, or a sequence with one per run. See
        [Evaluating in Parallel](../running/parallel.md) for a
        walkthrough.

        Args:
            config:               The configuration, or one per run.
            x0:                   The initial vector, or one per row.
            function:             The evaluation function, or one per run.
            handlers:             Optional handlers, fed by every run.
            report:               Optional callback, shared or one per run.
            limit:                The maximum number of runs at once.
            constraint_tolerance: The tolerance within which a constraint holds.
            bundle_size:          Evaluations per worker task, shared or one per
                                  run.
            metadata:             Optional dictionary attached to every result.
            f0:                   Optional function results at `x0`, shared or
                                  one per run.
            g0:                   Optional gradient results at `x0`, shared or
                                  one per run.
            report_gradients:     Whether `report` also receives gradient results.

        Returns:
            One [`OptimizationResult`][ropt.OptimizationResult] per run.

        Raises:
            RunsFailedError: If any of the runs raised.
            ValueError:      If `x0` has the wrong shape, or the sequences
                             given per run disagree in length.
            WorkflowError:   If this pool's session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _optimize_many(
            self._session,
            self._live_executor,
            config,
            x0,
            function,
            handlers=handlers,
            report=report,
            limit=limit,
            constraint_tolerance=constraint_tolerance,
            bundle_size=bundle_size,
            metadata=metadata,
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )

    def evaluate(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any],
        variables: ArrayLike,
        function: EvaluationFunction,
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | None = None,
        bundle_size: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult[FunctionResults | None]:
        """Evaluate a single variable vector on this pool, without optimizing.

        See [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            config:      The optimization configuration.
            variables:   The variable vector to evaluate.
            function:    The per-realization evaluation function.
            handlers:    Optional handlers, called in the order listed.
            report:      Optional callback invoked with the results.
            bundle_size: Evaluations per worker task, `None` for the pool's own.
            metadata:    Optional dictionary attached to the results.

        Returns:
            An [`EvaluationResult`][ropt.EvaluationResult] whose
            `results` is the [`FunctionResults`][ropt.results.FunctionResults]
            for the vector, or `None` if the evaluation was aborted.

        Raises:
            ValueError:    If `variables` is not a single vector.
            WorkflowError: If this pool's session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _evaluate(
            self._session,
            self._live_executor,
            config,
            variables,
            function,
            handlers=handlers,
            report=report,
            bundle_size=bundle_size,
            metadata=metadata,
        )

    def evaluate_batch(  # ruff: ignore[too-many-arguments]
        self,
        config: dict[str, Any],
        variables: ArrayLike,
        function: EvaluationFunction,
        *,
        handlers: Sequence[EventHandler] | None = None,
        report: ReportCallback | None = None,
        bundle_size: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult[tuple[FunctionResults, ...]]:
        """Evaluate a batch of variable vectors on this pool, without optimizing.

        Each row of `variables` is one vector, and the results come back in the
        same order. See [Running Optimizations](../running/running.md) for a
        walkthrough.

        Args:
            config:      The optimization configuration.
            variables:   The variable vectors to evaluate, one per row.
            function:    The per-realization evaluation function.
            handlers:    Optional handlers, called in the order listed.
            report:      Optional callback invoked with each evaluation.
            bundle_size: Evaluations per worker task, `None` for the pool's own.
            metadata:    Optional dictionary attached to every result.

        Returns:
            An [`EvaluationResult`][ropt.EvaluationResult] whose
            `results` holds one
            [`FunctionResults`][ropt.results.FunctionResults] per vector, and is
            empty if the batch was aborted.

        Raises:
            ValueError:    If `variables` is not a 2-D matrix.
            WorkflowError: If this pool's session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _evaluate_batch(
            self._session,
            self._live_executor,
            config,
            variables,
            function,
            handlers=handlers,
            report=report,
            bundle_size=bundle_size,
            metadata=metadata,
        )

    @overload
    def offload[T](self, work: Callable[[], T]) -> T: ...

    @overload
    def offload[T](self, work: Sequence[Callable[[], T]]) -> tuple[T, ...]: ...

    def offload[T](
        self, work: Callable[[], T] | Sequence[Callable[[], T]]
    ) -> T | tuple[T, ...]:
        """Run one or more arbitrary callables on this pool's workers.

        Pass a single zero-argument callable to run one call and get its result,
        or a sequence to run them concurrently and get a tuple of results in the
        order of `work`. Bind arguments with `functools.partial`. The callables
        must be picklable for a process or job pool.

        See [Running Optimizations](../running/running.md) for a walkthrough.

        Args:
            work: A single zero-argument callable, or a sequence of them.

        Returns:
            The single result, or a tuple of results in the order of `work`.

        Raises:
            AbortedError:   If an abort kept one of the callables from running.
            ExecutionError: If the machinery could not run a call.
            WorkflowError:  If this pool's session has closed.
        """  # ruff: ignore[docstring-extraneous-exception]
        return _offload(self._session, self._live_executor, work)
