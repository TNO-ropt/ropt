from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from ropt.components.compute_steps import OptimizationStep
from ropt.components.event_handlers import (
    CallbackHandler,
    EventDispatcher,
    EventForwardHandler,
    ResultsHandler,
)
from ropt.context import EnOptContext
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.exceptions import WorkflowError

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


def _event(context: EnOptContext) -> EnOptEvent:
    return EnOptEvent(event_type=EnOptEventType.FINISHED_EVALUATION, context=context)


def test_event_dispatcher_not_running_before_start() -> None:
    assert not EventDispatcher()._running.is_set()  # ruff: ignore[private-member-access]


def test_event_dispatcher_dispatch_before_start_raises(config: dict[str, Any]) -> None:
    event = _event(EnOptContext.model_validate(config))
    with pytest.raises(WorkflowError, match="not running"):
        EventDispatcher().dispatch_event(event)


@pytest.mark.asyncio
async def test_event_dispatcher_dispatch_after_stop_raises(
    config: dict[str, Any],
) -> None:
    event = _event(EnOptContext.model_validate(config))
    dispatcher = EventDispatcher()
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.cancel()
    assert not dispatcher._running.is_set()  # ruff: ignore[private-member-access]
    with pytest.raises(WorkflowError, match="not running"):
        dispatcher.dispatch_event(event)


@pytest.mark.asyncio
async def test_event_dispatcher_running_after_start() -> None:
    dispatcher = EventDispatcher()
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        assert dispatcher._running.is_set()  # ruff: ignore[private-member-access]
        dispatcher.cancel()
    assert not dispatcher._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.asyncio
async def test_event_dispatcher_already_running_raises() -> None:
    dispatcher = EventDispatcher()
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        with pytest.raises(WorkflowError, match="already running"):
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
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        await asyncio.to_thread(dispatcher.dispatch_event, event)
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
    matching = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        await asyncio.to_thread(
            dispatcher.dispatch_event,
            EnOptEvent(event_type=EnOptEventType.START_OPTIMIZER, context=context),
        )
        await asyncio.to_thread(dispatcher.dispatch_event, matching)
        await asyncio.to_thread(
            dispatcher.dispatch_event,
            EnOptEvent(event_type=EnOptEventType.FINISHED_OPTIMIZER, context=context),
        )
        dispatcher.cancel()
    assert received == [matching]


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
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        await asyncio.to_thread(dispatcher.dispatch_event, event)
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
    events = [_event(context) for _ in range(5)]
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        for event in events:
            await asyncio.to_thread(dispatcher.dispatch_event, event)
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
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        await asyncio.to_thread(forward.handle_event, event)
        dispatcher.cancel()
    assert received == [event]


@pytest.mark.asyncio
async def test_event_dispatcher_run_in_thread_dispatches(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)

    # A barrier proves the two thread handlers run concurrently: if they ran
    # sequentially the barrier would never release and the test would hang
    # (caught by the timeout mark).
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
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        await asyncio.to_thread(dispatcher.dispatch_event, event)
        dispatcher.cancel()
    assert received_a == [event]
    assert received_b == [event]


@pytest.mark.asyncio
async def test_event_dispatcher_rejects_an_event_handed_over_after_it_stopped(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    dispatcher = EventDispatcher()
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.cancel()
    with pytest.raises(WorkflowError, match="stopped"):
        await dispatcher._dispatch(_event(context))  # ruff: ignore[private-member-access]


class _FatalHandlerError(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_thread", [False, True])
async def test_handler_dispatching_on_own_dispatcher_raises(
    config: dict[str, Any], *, run_in_thread: bool
) -> None:
    # Events are handled one at a time, so a nested dispatch waits for an event
    # that cannot be processed until the handler returns: a silent deadlock.
    context = EnOptContext.model_validate(config)
    dispatcher = EventDispatcher()
    errors: list[BaseException] = []

    def _dispatch_again(event: EnOptEvent) -> None:  # ruff: ignore[unused-function-argument]
        try:
            dispatcher.dispatch_event(_event(context))
        except BaseException as exc:  # ruff: ignore[blind-except]
            errors.append(exc)

    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=_dispatch_again,
        ),
        run_in_thread=run_in_thread,
    )
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        await asyncio.to_thread(dispatcher.dispatch_event, _event(context))
        dispatcher.cancel()
    assert len(errors) == 1
    assert isinstance(errors[0], WorkflowError)
    assert "own dispatcher" in str(errors[0])


def _blocking_handler(
    busy: threading.Event, release: threading.Event, seen: list[int]
) -> CallbackHandler:
    def _handler(event: EnOptEvent) -> None:  # ruff: ignore[unused-function-argument]
        seen.append(len(seen))
        if len(seen) == 1:
            busy.set()
            release.wait(timeout=5)

    return CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_handler
    )


@pytest.mark.asyncio
async def test_events_queued_at_stop_are_handled(
    config: dict[str, Any],
) -> None:
    # cancel() queues a sentinel; events that arrive behind it must still be
    # drained, or their emitters are told the dispatcher stopped instead.
    context = EnOptContext.model_validate(config)
    busy, release = threading.Event(), threading.Event()
    seen: list[int] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        _blocking_handler(busy, release, seen), run_in_thread=True
    )
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        pending = [asyncio.create_task(dispatcher._dispatch(_event(context)))]  # ruff: ignore[private-member-access]
        await asyncio.to_thread(busy.wait)
        dispatcher.cancel()
        pending += [
            asyncio.create_task(dispatcher._dispatch(_event(context)))  # ruff: ignore[private-member-access]
            for _ in range(2)
        ]
        # One yield is enough: the cancel callback was queued first, so the
        # sentinel lands ahead of these two.
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*pending)
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_events_queued_at_failure_are_rejected(
    config: dict[str, Any],
) -> None:
    # The queue cannot simply be dropped when processing dies: every emitter is
    # blocked on its own event and would otherwise wait forever.
    context = EnOptContext.model_validate(config)
    busy, release = threading.Event(), threading.Event()
    seen: list[int] = []

    def _explode(event: EnOptEvent) -> None:  # ruff: ignore[unused-function-argument]
        seen.append(len(seen))
        busy.set()
        release.wait(timeout=5)
        msg = "handler exploded"
        raise _FatalHandlerError(msg)

    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_explode
        ),
        run_in_thread=True,
    )
    pending: list[asyncio.Task[None]] = []
    with pytest.raises(BaseExceptionGroup):  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await dispatcher.start(tg)
            pending.append(asyncio.create_task(dispatcher._dispatch(_event(context))))  # ruff: ignore[private-member-access]
            await asyncio.to_thread(busy.wait)
            pending += [
                asyncio.create_task(dispatcher._dispatch(_event(context)))  # ruff: ignore[private-member-access]
                for _ in range(2)
            ]
            await asyncio.sleep(0)
            release.set()
    outcomes = await asyncio.gather(*pending, return_exceptions=True)
    assert seen == [0]
    stopped = [exc for exc in outcomes[1:] if isinstance(exc, WorkflowError)]
    assert len(stopped) == 2
    assert all("stopped" in str(exc) for exc in stopped)


@pytest.mark.asyncio
async def test_handler_base_exception_reaches_emitter(
    config: dict[str, Any],
) -> None:
    # A BaseException is fatal, but reporting it as "dispatcher stopped" would
    # hide the only description of what actually went wrong.
    def _raise_fatal(event: EnOptEvent) -> None:  # ruff: ignore[unused-function-argument]
        msg = "handler exploded"
        raise _FatalHandlerError(msg)

    context = EnOptContext.model_validate(config)
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_raise_fatal
        )
    )
    with pytest.raises(BaseExceptionGroup) as excinfo:  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await dispatcher.start(tg)
            with pytest.raises(_FatalHandlerError, match="handler exploded"):
                await asyncio.to_thread(dispatcher.dispatch_event, _event(context))
    matched, _ = excinfo.value.split(_FatalHandlerError)
    assert matched is not None


def test_event_dispatcher_reports_a_closed_loop_as_a_workflow_error(
    config: dict[str, Any],
) -> None:
    # A caller whose loop is already gone must get the documented error rather
    # than a bare RuntimeError from asyncio.
    context = EnOptContext.model_validate(config)
    dispatcher = EventDispatcher()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
    dispatcher._loop = loop  # ruff: ignore[private-member-access]
    dispatcher._running.set()  # ruff: ignore[private-member-access]
    with pytest.raises(WorkflowError, match="stopped"):
        dispatcher.dispatch_event(_event(context))


def test_cancelling_dispatcher_without_loop() -> None:
    dispatcher = EventDispatcher()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
    dispatcher._loop = loop  # ruff: ignore[private-member-access]
    dispatcher._queue = asyncio.Queue()  # ruff: ignore[private-member-access]
    dispatcher.cancel()


@pytest.mark.asyncio
async def test_event_dispatcher_threaded_handler_avoids_the_shared_default_pool(
    config: dict[str, Any],
) -> None:
    # Occupy asyncio's shared default executor completely: a threaded handler
    # dispatched through it (the old behavior) could not run at all.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release = threading.Event()
    occupied = loop.run_in_executor(None, release.wait)

    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        ),
        run_in_thread=True,
    )
    event = _event(context)
    finished = asyncio.Event()

    def _dispatch() -> None:
        dispatcher.dispatch_event(event)
        loop.call_soon_threadsafe(finished.set)

    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        threading.Thread(target=_dispatch, daemon=True).start()
        await finished.wait()
        dispatcher.cancel()
    release.set()
    await occupied
    assert received == [event]


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_thread", [True, False])
async def test_event_dispatcher_handler_pool_is_lazy_and_shut_down(
    config: dict[str, Any], run_in_thread: Any
) -> None:
    context = EnOptContext.model_validate(config)
    received: list[EnOptEvent] = []
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        ),
        run_in_thread=run_in_thread,
    )
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        assert dispatcher._thread_pool is None  # ruff: ignore[private-member-access]
        await asyncio.to_thread(dispatcher.dispatch_event, event)
        assert (
            dispatcher._thread_pool is not None  # ruff: ignore[private-member-access]
        ) is run_in_thread
        dispatcher.cancel()
    assert received == [event]
    assert dispatcher._thread_pool is None  # ruff: ignore[private-member-access]


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
async def test_event_dispatcher_reraises_failing_handler(
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
        )
    )
    dispatcher.add_event_handler(
        CallbackHandler(
            event_types={EnOptEventType.FINISHED_EVALUATION},
            callback=received.append,
        )
    )
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        # The failure surfaces on the caller's own stack, unwrapped.
        with pytest.raises(_HandlerError, match=_ERROR_MESSAGE):
            await asyncio.to_thread(dispatcher.dispatch_event, event)
        dispatcher.cancel()
    # Every handler for the event still ran before the error surfaced.
    assert received == [event]


@pytest.mark.asyncio
async def test_event_dispatcher_reraises_failing_thread_handler(
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
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        with pytest.raises(_HandlerError, match=_ERROR_MESSAGE):
            await asyncio.to_thread(dispatcher.dispatch_event, event)
        dispatcher.cancel()
    assert received == [event]


@pytest.mark.asyncio
async def test_event_forward_handler_reraises_failure(
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
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        # The forward handler re-raises the original on the emitting run's stack.
        with pytest.raises(_HandlerError, match=_ERROR_MESSAGE):
            await asyncio.to_thread(forward.handle_event, event)
        dispatcher.cancel()


@pytest.mark.asyncio
async def test_event_dispatcher_base_exception_tears_down(
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
    event = _event(context)

    # A BaseException is not isolated: it stays the teardown backstop and tears
    # the dispatcher task group down as a BaseExceptionGroup.
    async def _run() -> None:
        async with asyncio.TaskGroup() as tg:
            await dispatcher.start(tg)
            tg.create_task(asyncio.to_thread(dispatcher.dispatch_event, event))

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await _run()
    matched, _ = exc_info.value.split(_HandlerBaseError)
    assert matched is not None


def test_dispatcher_owned_handler_rejects_concurrent_handle_event(
    config: dict[str, Any],
) -> None:
    # The dispatcher serializes calls, but the concurrency guard still applies to
    # a dispatcher-owned handler, so a stray direct call entering handle_event
    # while it is already running (bypassing the dispatcher) is rejected.
    event = _event(EnOptContext.model_validate(config))
    entered = threading.Event()
    release = threading.Event()

    def _block(_event: EnOptEvent) -> None:
        entered.set()
        release.wait()

    handler = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_block
    )
    EventDispatcher().add_event_handler(handler)

    first = threading.Thread(target=handler.handle_event, args=(event,))
    first.start()
    entered.wait()
    try:
        with pytest.raises(WorkflowError, match="already running on another thread"):
            handler.handle_event(event)
    finally:
        release.set()
        first.join()


@pytest.mark.asyncio
async def test_event_dispatcher_removed_handler_no_longer_receives_events(
    config: dict[str, Any],
) -> None:
    context = EnOptContext.model_validate(config)
    received_a: list[EnOptEvent] = []
    received_b: list[EnOptEvent] = []
    handler_a = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=received_a.append
    )
    handler_b = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=received_b.append
    )
    dispatcher = EventDispatcher()
    dispatcher.add_event_handler(handler_a)
    dispatcher.add_event_handler(handler_b)
    event = _event(context)
    async with asyncio.TaskGroup() as tg:
        await dispatcher.start(tg)
        dispatcher.remove_event_handler(handler_a)
        await asyncio.to_thread(dispatcher.dispatch_event, event)
        dispatcher.cancel()
    assert received_a == []
    assert received_b == [event]


def test_event_dispatcher_removed_handler_can_be_added_to_another_dispatcher() -> None:
    received: list[EnOptEvent] = []
    handler = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=received.append
    )
    first = EventDispatcher()
    first.add_event_handler(handler)
    first.remove_event_handler(handler)
    EventDispatcher().add_event_handler(handler)


def test_event_dispatcher_remove_unknown_handler_raises() -> None:
    received: list[EnOptEvent] = []
    handler = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=received.append
    )
    with pytest.raises(WorkflowError, match="not added to the dispatcher"):
        EventDispatcher().remove_event_handler(handler)


def test_event_dispatcher_remove_returns_run_in_thread_flag() -> None:
    dispatcher = EventDispatcher()
    threaded = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=lambda _event: None
    )
    plain = CallbackHandler(
        event_types={EnOptEventType.FINISHED_EVALUATION}, callback=lambda _event: None
    )
    dispatcher.add_event_handler(threaded, run_in_thread=True)
    dispatcher.add_event_handler(plain)
    assert dispatcher.remove_event_handler(threaded) is True
    assert dispatcher.remove_event_handler(plain) is False
