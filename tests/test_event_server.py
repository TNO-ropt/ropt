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
    EventForwardHandler,
    ResultsHandler,
)
from ropt.workflow.executors import EventServer

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


def test_event_server_not_running_before_start() -> None:
    assert not EventServer().is_running()


@pytest.mark.asyncio
async def test_event_server_running_after_start() -> None:
    server = EventServer()
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        assert server.is_running()
        server.cancel()
    assert not server.is_running()


@pytest.mark.asyncio
async def test_event_server_already_running_raises() -> None:
    server = EventServer()
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        with pytest.raises(RuntimeError, match="already running"):
            await server.start(tg)
        server.cancel()


@pytest.mark.asyncio
async def test_event_server_dispatches_to_handler(config: dict[str, Any]) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    server = EventServer()
    server.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        server.put_event(event)
        server.cancel()
    assert received == [event]


@pytest.mark.asyncio
async def test_event_server_filters_by_event_type(config: dict[str, Any]) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    server = EventServer()
    server.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        server.put_event(
            EnOptEvent(event_type=EnOptEventType.START_OPTIMIZER, context=context)
        )
        server.put_event(
            EnOptEvent(event_type=EnOptEventType.START_EVALUATION, context=context)
        )
        server.put_event(
            EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
        )
        server.put_event(
            EnOptEvent(event_type=EnOptEventType.FINISHED_OPTIMIZER, context=context)
        )
        server.cancel()
    assert len(received) == 1
    assert received[0].event_type == EnOptEventType.FINISHED_EVALUATION


@pytest.mark.asyncio
async def test_event_server_multiple_handlers_all_receive(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    received_a: list[EnOptEvent] = []
    received_b: list[EnOptEvent] = []
    server = EventServer()
    server.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received_a.append,
        )
    )
    server.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received_b.append,
        )
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        server.put_event(event)
        server.cancel()
    assert received_a == [event]
    assert received_b == [event]


@pytest.mark.asyncio
async def test_event_server_events_processed_in_order(config: dict[str, Any]) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    server = EventServer()
    server.add_event_handler(
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
        await server.start(tg)
        for event in events:
            server.put_event(event)
        server.cancel()
    assert received == events


def test_event_forward_handler_event_types() -> None:
    server = EventServer()
    forward = EventForwardHandler(
        server, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    assert forward.event_types == {EnOptEventType.FINISHED_EVALUATION}


@pytest.mark.asyncio
async def test_event_forward_handler_forwards_to_server(config: dict[str, Any]) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    server = EventServer()
    server.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    forward = EventForwardHandler(
        server, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        forward.handle_event(event)
        server.cancel()
    assert received == [event]


@pytest.mark.asyncio
async def test_event_server_run_in_thread_dispatches(config: dict[str, Any]) -> None:
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

    server = EventServer()
    server.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_handler_a
        ),
        run_in_thread=True,
    )
    server.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_handler_b
        ),
        run_in_thread=True,
    )
    event = EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        server.put_event(event)
        server.cancel()
    assert received_a == [event]
    assert received_b == [event]


@pytest.mark.asyncio
async def test_event_server_with_optimization_step(
    config: dict[str, Any], evaluator: Any
) -> None:
    event_server = EventServer()
    result_handler = ResultsHandler()
    event_server.add_event_handler(result_handler)

    step = OptimizationStep(evaluator=evaluator())
    step.add_event_handler(
        EventForwardHandler(
            event_server, event_types={EnOptEventType.FINISHED_EVALUATION}
        )
    )

    async with asyncio.TaskGroup() as tg:
        await event_server.start(tg)
        context = EnOptContext.model_validate(config)
        await asyncio.to_thread(step.run, variables=initial_values, context=context)
        event_server.cancel()

    assert result_handler["results"] is not None
    assert np.allclose(
        result_handler["results"].evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
