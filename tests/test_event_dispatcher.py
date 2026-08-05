from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
import pytest

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.workflow.compute_steps import OptimizationStep
from ropt.workflow.event_handlers import (
    CallbackHandler,
    EventDispatcher,
    EventForwardHandler,
    ResultsHandler,
)

pytestmark = pytest.mark.timeout(5)

initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {"max_functions": 20},
        "backend": {"convergence_tolerance": 1e-5},
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {"weights": [0.75, 0.25]},
    }


def test_event_dispatcher_not_running_before_start() -> None:
    assert not EventDispatcher().is_running()


def test_event_dispatcher_put_event_before_start_raises(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    with pytest.raises(RuntimeError, match="not running"):
        EventDispatcher().put_event(event)


@pytest.mark.asyncio
async def test_event_dispatcher_put_event_after_stop_raises(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    dispatcher = EventDispatcher()
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.cancel()
    assert not dispatcher.is_running()
    with pytest.raises(RuntimeError, match="not running"):
        dispatcher.put_event(event)


@pytest.mark.asyncio
async def test_event_dispatcher_running_after_start() -> None:
    dispatcher = EventDispatcher()
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        assert dispatcher.is_running()
        dispatcher.cancel()
    assert not dispatcher.is_running()


@pytest.mark.asyncio
async def test_event_dispatcher_already_running_raises() -> None:
    dispatcher = EventDispatcher()
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        with pytest.raises(RuntimeError, match="already running"):
            await dispatcher.start(tg)
        dispatcher.cancel()


@pytest.mark.asyncio
async def test_event_dispatcher_dispatches_to_handler(config: dict[str, Any]) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.put_event(event)
        dispatcher.cancel()
    assert received == [event]


@pytest.mark.asyncio
async def test_event_dispatcher_filters_by_event_type(config: dict[str, Any]) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.put_event(
            EnOptEvent(event_type=EnOptEventType.START_OPTIMIZER, context=context)
        )
        dispatcher.put_event(
            EnOptEvent(event_type=EnOptEventType.START_EVALUATION, context=context)
        )
        dispatcher.put_event(
            EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
        )
        dispatcher.put_event(
            EnOptEvent(event_type=EnOptEventType.FINISHED_OPTIMIZER, context=context)
        )
        dispatcher.cancel()
    assert len(received) == 1
    assert received[0].event_type == EnOptEventType.FINISHED_EVALUATION


@pytest.mark.asyncio
async def test_event_dispatcher_multiple_handlers_all_receive(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    received_a: list[EnOptEvent] = []
    received_b: list[EnOptEvent] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received_a.append,
        )
    )
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received_b.append,
        )
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.put_event(event)
        dispatcher.cancel()
    assert received_a == [event]
    assert received_b == [event]


@pytest.mark.asyncio
async def test_event_dispatcher_events_processed_in_order(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    events = [
        EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
        for _ in range(5)
    ]
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        for event in events:
            dispatcher.put_event(event)
        dispatcher.cancel()
    assert received == events


def test_event_forward_handler_event_types() -> None:
    dispatcher = EventDispatcher()
    forward = EventForwardHandler(
        dispatcher, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    assert forward.event_types == {EnOptEventType.FINISHED_EVALUATION}


@pytest.mark.asyncio
async def test_event_forward_handler_forwards_to_dispatcher(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    forward = EventForwardHandler(
        dispatcher, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        forward.handle_event(event)
        dispatcher.cancel()
    assert received == [event]


@pytest.mark.asyncio
async def test_event_dispatcher_run_in_thread_dispatches(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)

    # Use a barrier to prove the two thread handlers run concurrently: if they
    # ran sequentially the barrier would never be reached by both threads and
    # the test would hang (caught by the timeout mark).
    barrier = threading.Barrier(2)
    received_a: list[EnOptEvent] = []
    received_b: list[EnOptEvent] = []

    def _handler_a(event: EnOptEvent) -> None:
        barrier.wait()
        received_a.append(event)

    def _handler_b(event: EnOptEvent) -> None:
        barrier.wait()
        received_b.append(event)

    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_handler_a
        ),
        run_in_thread=True,
    )
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_handler_b
        ),
        run_in_thread=True,
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.put_event(event)
        dispatcher.cancel()
    assert received_a == [event]
    assert received_b == [event]


@pytest.mark.asyncio
async def test_event_dispatcher_with_optimization_step(
    config: dict[str, Any], evaluator: Any
) -> None:
    event_dispatcher = EventDispatcher()
    result_handler = ResultsHandler()
    event_dispatcher.add_event_handler(result_handler)

    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(
        EventForwardHandler(
            event_dispatcher, event_types={EnOptEventType.FINISHED_EVALUATION}
        )
    )

    async with asyncio.TaskGroup() as tg:
        await event_dispatcher.start(tg)
        context = EnOptContext.model_validate(config)
        await asyncio.to_thread(step.run, variables=initial_values, context=context)
        event_dispatcher.cancel()

    assert result_handler["results"] is not None
    assert np.allclose(
        result_handler["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


class _HandlerError(Exception):
    pass


class _HandlerBaseError(BaseException):
    pass


_ERROR_MESSAGE = "boom"
_FATAL_MESSAGE = "fatal"


@pytest.mark.asyncio
async def test_event_dispatcher_isolates_failing_handler(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    calls: list[EnOptEvent] = []
    received: list[EnOptEvent] = []

    def _fail(event: EnOptEvent) -> None:
        calls.append(event)
        raise _HandlerError(_ERROR_MESSAGE)

    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_fail
        )
    )
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    # Exiting the task group without a BaseExceptionGroup proves the handler
    # fault did not escape the dispatcher task and tear the session down.
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.put_event(event)
        dispatcher.put_event(event)
        dispatcher.cancel()
    # The healthy handler received both events and the failing handler was
    # invoked for both (recorded each time, not quarantined).
    assert received == [event, event]
    assert calls == [event, event]
    # The failure is recorded and can be re-raised on the caller's stack.
    with pytest.raises(_HandlerError, match=_ERROR_MESSAGE):
        dispatcher.raise_pending_error()


@pytest.mark.asyncio
async def test_event_dispatcher_isolates_failing_thread_handler(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []

    def _fail(_event: EnOptEvent) -> None:
        raise _HandlerError(_ERROR_MESSAGE)

    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_fail
        ),
        run_in_thread=True,
    )
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.put_event(event)
        dispatcher.cancel()
    # A run_in_thread handler is isolated exactly like an inline one.
    assert received == [event]
    with pytest.raises(_HandlerError, match=_ERROR_MESSAGE):
        dispatcher.raise_pending_error()


@pytest.mark.asyncio
async def test_event_forward_handler_reraises_recorded_failure(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)

    def _fail(_event: EnOptEvent) -> None:
        raise _HandlerError(_ERROR_MESSAGE)

    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_fail
        )
    )
    forward = EventForwardHandler(
        dispatcher, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    forward.register_compute_step()
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.put_event(event)
        dispatcher.cancel()
    # On the emitting run's own stack the forward handler re-raises the original
    # exception, unwrapped — a clean stop, not a BaseExceptionGroup.
    with pytest.raises(_HandlerError, match=_ERROR_MESSAGE):
        forward.handle_event(event)


@pytest.mark.asyncio
async def test_event_dispatcher_base_exception_propagates(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)

    def _fail(_event: EnOptEvent) -> None:
        raise _HandlerBaseError(_FATAL_MESSAGE)

    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_fail
        )
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)

    async def _run() -> None:
        async with asyncio.TaskGroup() as tg:
            await dispatcher.start(tg)
            dispatcher.put_event(event)
            dispatcher.cancel()

    # A BaseException is not isolated: it stays the teardown backstop and tears
    # the session task group down as a BaseExceptionGroup.
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await _run()
    _, unhandled = exc_info.value.split(_HandlerBaseError)
    assert unhandled is None
