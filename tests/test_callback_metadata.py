from __future__ import annotations

import threading
from functools import partial
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

from ropt import (
    CallbackHandler,
    EvaluationFunctionResult,
    HistoryHandler,
    evaluate,
    evaluate_batch,
    optimize,
    run,
    session,
)
from ropt._core import EnsembleOptimizer
from ropt.enums import EnOptEventType, ExitCode
from ropt.results import FunctionResults, GradientResults

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from ropt import (
        EvaluationFunction,
        EvaluationFunctionContext,
        EventHandler,
        WorkerPool,
    )
    from ropt.events import EnOptEvent
    from ropt.results import Results

_CONFIG: dict[str, object] = {
    "optimizer": {"max_functions": 2},
    "backend": {"method": "slsqp"},
    "variables": {
        "variable_count": 1,
        "perturbation_magnitudes": 0.01,
        "seed": 11,
    },
    "realizations": {"weights": [1.0, 1.0]},
    "gradient": {"number_of_perturbations": 2},
}
_Method = Literal["optimize", "evaluate", "evaluate_batch"]


def _objective(
    variables: NDArray[np.float64], _context: EvaluationFunctionContext
) -> float:
    return float(np.sum((variables - 1.0) ** 2))


def _record_metadata(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    assert context.metadata is not None
    return EvaluationFunctionResult(
        objectives=_objective(variables, context),
        metadata={
            "run": context.metadata["run"],
            "prepared": context.metadata["prepared"],
        },
    )


def _handler(callback: Callable[[EnOptEvent], None]) -> CallbackHandler:
    return CallbackHandler(
        event_types={EnOptEventType.START_EVALUATION}, callback=callback
    )


def _run_with_metadata(
    method: _Method,
    function: EvaluationFunction,
    handlers: Sequence[EventHandler],
    *,
    metadata: dict[str, object] | None = None,
    pool: WorkerPool | None = None,
) -> tuple[Results, ...]:
    history = HistoryHandler()
    ordered = [*handlers, history]
    if method == "optimize":
        optimize(
            _CONFIG, [0.0], function, handlers=ordered, metadata=metadata, pool=pool
        )
    elif method == "evaluate":
        evaluate(
            _CONFIG, [0.0], function, handlers=ordered, metadata=metadata, pool=pool
        )
    else:
        evaluate_batch(
            _CONFIG,
            [[0.0], [0.5]],
            function,
            handlers=ordered,
            metadata=metadata,
            pool=pool,
        )
    return history.results


@pytest.mark.parametrize("method", ["optimize", "evaluate", "evaluate_batch"])
@pytest.mark.parametrize(
    "pool_kind",
    ["none", "thread", pytest.param("process", marks=pytest.mark.slow)],
)
@pytest.mark.timeout(60)
def test_callback_metadata_reaches_evaluations_and_results(
    method: _Method, pool_kind: str
) -> None:
    def prepare(event: EnOptEvent) -> None:
        run_id = event.metadata["run"]
        assert isinstance(run_id, int)
        event.metadata["prepared"] = run_id + 4

    with session() as opened:
        pool = None
        if pool_kind == "thread":
            pool = opened.thread_pool(workers=2)
        elif pool_kind == "process":
            pool = opened.process_pool(workers=2)
        results = _run_with_metadata(
            method,
            _record_metadata,
            [_handler(prepare)],
            metadata={"run": 3},
            pool=pool,
        )

    assert results
    for result in results:
        assert result.metadata == {"run": 3, "prepared": 7}
        assert isinstance(result, FunctionResults | GradientResults)
        assert np.all(result.evaluations.metadata["run"] == 3)
        assert np.all(result.evaluations.metadata["prepared"] == 7)


def test_callback_metadata_shares_changes_between_handlers() -> None:
    original: dict[str, object] = {"keep": 1, "replace": 0, "nested": {"old": 1}}
    updates = {"nested": {"new": 2}}
    handler_metadata: list[dict[str, object]] = []

    def first(event: EnOptEvent) -> None:
        assert event.metadata == original
        handler_metadata.append(event.metadata)
        event.metadata.update({"replace": 1, "first": True})
        event.metadata.update(updates)
        updates["nested"]["new"] = 99
        assert event.metadata["nested"] == {"new": 99}

    def second(event: EnOptEvent) -> None:
        assert event.metadata is handler_metadata[0]
        assert event.metadata["replace"] == 1
        assert event.metadata["first"] is True
        event.metadata.update({"replace": 2, "second": True})

    seen: list[dict[str, object] | None] = []

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        seen.append(context.metadata)
        return _objective(variables, context)

    results = _run_with_metadata(
        "evaluate", objective, [_handler(first), _handler(second)], metadata=original
    )
    expected = {
        "keep": 1,
        "replace": 2,
        "nested": {"new": 99},
        "first": True,
        "second": True,
    }
    assert seen == [expected, expected]
    assert results[0].metadata == expected
    assert original == {"keep": 1, "replace": 0, "nested": {"old": 1}}
    updates["nested"]["new"] = 100
    assert results[0].metadata == expected


def test_callback_metadata_reassignment_takes_effect() -> None:
    def prepare(event: EnOptEvent) -> None:
        event.metadata = {**event.metadata, "prepared": True}

    results = _run_with_metadata(
        "evaluate", _objective, [_handler(prepare)], metadata={"run": 3}
    )
    assert results[0].metadata == {"run": 3, "prepared": True}


@pytest.mark.parametrize("with_metadata", [False, True])
def test_callback_metadata_resets_between_batches(
    monkeypatch: pytest.MonkeyPatch, *, with_metadata: bool
) -> None:
    baseline_values = [1]
    original: dict[str, object] | None = (
        {"nested": baseline_values, "replace": "baseline", "removed": True}
        if with_metadata
        else None
    )
    baseline = (
        {"nested": [1], "replace": "baseline", "removed": True}
        if with_metadata
        else None
    )
    starts = 0
    seen: list[EvaluationFunctionContext] = []

    def prepare(event: EnOptEvent) -> None:
        nonlocal starts
        starts += 1
        assert event.metadata == (baseline or {})
        if starts == 1:
            event.metadata.update(
                {"nested": [2], "replace": "batch", "temporary": True}
            )
            if with_metadata:
                del event.metadata["removed"]
            baseline_values.append(99)
        elif starts == 3:
            event.metadata["combined"] = True

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        seen.append(context)
        return _objective(variables, context)

    def start(optimizer: EnsembleOptimizer, variables: NDArray[np.float64]) -> ExitCode:
        evaluator = optimizer._function_evaluator  # ruff: ignore[private-member-access]
        first = evaluator.calculate(
            variables, compute_functions=True, compute_gradients=False
        )
        cached = evaluator.calculate(
            variables, compute_functions=True, compute_gradients=False
        )
        assert cached[0] is first[0]
        assert starts == 1
        evaluator.calculate(variables, compute_functions=False, compute_gradients=True)
        evaluator.calculate(
            variables + 1.0, compute_functions=True, compute_gradients=True
        )
        return ExitCode.FINISHED

    monkeypatch.setattr(EnsembleOptimizer, "start", start)
    results = _run_with_metadata(
        "optimize", objective, [_handler(prepare)], metadata=original
    )

    assert starts == 3
    assert len(results) == 4
    expected = [
        {"nested": [2], "replace": "batch", "temporary": True},
        baseline,
        {**(baseline or {}), "combined": True},
    ]
    batch_ids = list(dict.fromkeys(context.batch_id for context in seen))
    assert len(batch_ids) == 3
    for context in seen:
        assert context.metadata == expected[batch_ids.index(context.batch_id)]
    for result in results:
        assert result.metadata == (expected[batch_ids.index(result.batch_id)] or {})


def test_callback_metadata_snapshots_are_independent() -> None:
    supplied = [1]
    updates = [2]

    def prepare(event: EnOptEvent) -> None:
        event.metadata["updated"] = updates

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        assert context.metadata is not None
        context.metadata["supplied"].append(99)
        context.metadata["updated"].append(99)
        return _objective(variables, context)

    results = _run_with_metadata(
        "evaluate_batch",
        objective,
        [_handler(prepare)],
        metadata={"supplied": supplied},
    )
    assert supplied == [1]
    assert updates == [2]
    assert all(
        result.metadata == {"supplied": [1], "updated": [2]} for result in results
    )
    results[0].metadata["supplied"].append(100)
    results[0].metadata["updated"].append(100)
    assert results[1].metadata == {"supplied": [1], "updated": [2]}


@pytest.mark.parametrize("with_metadata", [False, True])
def test_callback_metadata_can_clear_defaults(*, with_metadata: bool) -> None:
    original: dict[str, object] | None = {"run": 3} if with_metadata else None
    seen: list[dict[str, object] | None] = []

    def prepare(event: EnOptEvent) -> None:
        assert event.metadata == (original or {})
        event.metadata["temporary"] = True
        event.metadata.clear()

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        seen.append(context.metadata)
        return _objective(variables, context)

    results = _run_with_metadata(
        "evaluate", objective, [_handler(prepare)], metadata=original
    )
    expected: dict[str, object] | None = {} if with_metadata else None
    assert seen == [expected, expected]
    assert results[0].metadata == {}
    assert original == ({"run": 3} if with_metadata else None)


@pytest.mark.parametrize(
    "event_type",
    [item for item in EnOptEventType if item is not EnOptEventType.START_EVALUATION],
)
def test_callback_metadata_ignores_other_event_types(
    event_type: EnOptEventType,
) -> None:
    seen: list[dict[str, object] | None] = []

    def prepare(event: EnOptEvent) -> None:
        assert event.metadata == {}
        event.metadata["ignored"] = True

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        seen.append(context.metadata)
        return _objective(variables, context)

    handler = CallbackHandler(event_types={event_type}, callback=prepare)
    method: _Method = (
        "optimize"
        if event_type
        in {EnOptEventType.START_OPTIMIZER, EnOptEventType.FINISHED_OPTIMIZER}
        else "evaluate"
    )
    results = _run_with_metadata(method, objective, [handler])
    assert seen
    assert all(metadata is None for metadata in seen)
    assert all(result.metadata == {} for result in results)


def test_callback_metadata_later_edits_have_no_effect() -> None:
    events: list[EnOptEvent] = []
    nested_values: list[list[int]] = []

    def prepare(event: EnOptEvent) -> None:
        values = [len(events) + 1]
        event.metadata.update({"valid": True, "nested": values})
        events.append(event)
        nested_values.append(values)

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        events[-1].metadata["valid"] = False
        nested_values[-1].append(99)
        assert context.metadata is not None
        assert context.metadata["valid"] is True
        assert context.metadata["nested"] == [len(events)]
        return _objective(variables, context)

    results = _run_with_metadata("optimize", objective, [_handler(prepare)])
    assert len(events) > 1
    for event in events:
        event.metadata.clear()
    for result in results:
        assert result.metadata["valid"] is True
        assert len(result.metadata["nested"]) == 1


def test_callback_metadata_handler_error_prevents_evaluation() -> None:
    evaluated = False

    def prepare(event: EnOptEvent) -> None:
        event.metadata["prepared"] = True
        msg = "Preparation failed"
        raise ValueError(msg)

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        nonlocal evaluated
        evaluated = True
        return _objective(variables, context)

    with pytest.raises(ValueError, match="Preparation failed"):
        _run_with_metadata("evaluate", objective, [_handler(prepare)])
    assert not evaluated


def test_callback_metadata_does_not_use_callback_return_values() -> None:
    def prepare(event: EnOptEvent) -> bool:
        event.metadata["prepared"] = True
        return True

    callback = cast("Callable[[EnOptEvent], None]", prepare)
    results = _run_with_metadata("optimize", _objective, [_handler(callback)])
    assert len(results) > 1
    assert all(result.metadata == {"prepared": True} for result in results)


@pytest.mark.timeout(30)
def test_callback_metadata_isolated_with_a_shared_handler() -> None:
    barrier = threading.Barrier(2, timeout=10)
    prepared = 0

    def prepare(event: EnOptEvent) -> None:
        nonlocal prepared
        prepared += 1
        run_id = event.metadata["run"]
        assert isinstance(run_id, int)
        event.metadata["prepared"] = run_id + 1

    def objective(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        if context.row_idx == 0:
            barrier.wait()
        return _record_metadata(variables, context)

    handler = _handler(prepare)
    with session() as opened:
        results = run(
            [
                partial(
                    evaluate,
                    _CONFIG,
                    [0.0],
                    objective,
                    metadata={"run": run_id},
                    handlers=[handler],
                )
                for run_id in range(2)
            ],
            pool=opened.thread_pool(workers=2),
        )
    assert prepared == 2
    assert {outcome.results.metadata["prepared"] for outcome in results} == {1, 2}
    for run_id, outcome in enumerate(results):
        result = outcome.results
        assert result.metadata["run"] == run_id
        assert result.metadata["prepared"] == run_id + 1
        assert np.all(result.evaluations.metadata["run"] == run_id)
        assert np.all(
            result.evaluations.metadata["prepared"] == result.metadata["prepared"]
        )
