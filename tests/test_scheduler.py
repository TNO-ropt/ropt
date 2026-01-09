from __future__ import annotations

# ruff: noqa: SLF001, FBT001
import asyncio
import uuid
from collections.abc import Iterable
from typing import Any

import pytest

from ropt.scheduler import (
    ErrorEvent,
    Event,
    MainEventLoop,
    PollerTask,
    Scheduler,
    TaskBase,
    TaskEvent,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(1)]


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_task_done(is_background_task: bool) -> None:
    class MyTask(TaskBase):
        async def run(self) -> None:
            if not is_background_task:
                self.mark_finished()

    queue: asyncio.Queue[Event] = asyncio.Queue()
    task = MyTask()
    assert task.state == "pending"
    task.schedule(queue)

    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.state == "running"
    if is_background_task:
        task.mark_finished()
    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.state == "done"
    assert task.state == "done"  # type: ignore[comparison-overlap]
    assert task._task is not None
    assert task._task.done()


async def test_task_error() -> None:
    class MyTask(TaskBase):
        async def run(self) -> None:  # noqa: PLR6301
            msg = "test error"
            raise RuntimeError(msg)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    task = MyTask()
    assert task.state == "pending"
    task.schedule(queue)

    while True:
        event = await queue.get()
        assert isinstance(event, TaskEvent)
        if event.state == "done":
            assert isinstance(event.error, RuntimeError)
            assert str(event.error) == "test error"
            break

    assert task.state == "done"  # type: ignore[comparison-overlap]
    assert task._task is not None
    assert task._task.done()


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_task_cancel(is_background_task: bool) -> None:
    class MyTask(TaskBase):
        async def run(self) -> None:
            await asyncio.sleep(1)
            if not is_background_task:
                self.mark_finished()

    queue: asyncio.Queue[Event] = asyncio.Queue()
    task = MyTask()
    assert task.state == "pending"
    task.schedule(queue)

    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.state == "running"

    await task.cancel()

    if is_background_task:
        task.mark_finished()

    event = await queue.get()
    assert isinstance(event, TaskEvent)

    assert event.state == "cancelled"

    assert task._task is not None
    with pytest.raises(asyncio.CancelledError):
        await task._task


async def test_task_cancel_raises_error() -> None:
    class MyTask(TaskBase):
        async def run(self) -> None:  # noqa: PLR6301
            try:
                await asyncio.sleep(1)
            finally:
                msg = "error during cancel"
                raise RuntimeError(msg)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    task = MyTask().schedule(queue)

    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.state == "running"

    await task.cancel()

    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.state == "cancelled"
    assert str(event.error) == "error during cancel"
    assert isinstance(event.error, RuntimeError)

    # Task is done
    assert task._task is not None
    assert task._task.done()


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_task_with_semaphore(is_background_task: bool) -> None:
    sem = asyncio.Semaphore(1)
    queue: asyncio.Queue[Event] = asyncio.Queue()

    task_event = asyncio.Event()

    class TrackedTask(TaskBase):
        async def run(self) -> None:
            await task_event.wait()  # keep this running as long as we want
            if not is_background_task:
                self.mark_finished()

    task1 = TrackedTask().schedule(queue, semaphore=sem)
    task2 = TrackedTask().schedule(queue, semaphore=sem)

    event = await queue.get()
    assert event.source == task1.tid
    assert isinstance(event, TaskEvent)
    assert event.state == "running"
    assert task1.state == "running"
    assert task2.state == "pending"

    task_event.set()
    if is_background_task:
        task1.mark_finished()

    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.source == task1.tid
    assert event.state == "done"
    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.source == task2.tid
    assert event.state == "running"
    if is_background_task:
        task2.mark_finished()
    event = await queue.get()
    assert isinstance(event, TaskEvent)
    assert event.source == task2.tid
    assert event.state == "done"


class DummyTask(TaskBase):
    def __init__(
        self,
        *,
        dependencies: list[uuid.UUID] | None = None,
    ) -> None:
        super().__init__(tid=uuid.uuid4(), dependencies=dependencies)

    async def run(self) -> None:  # noqa: PLR6301
        await asyncio.sleep(0)


async def _do_not_submit() -> None:
    # patch _submit_ready_tasks with this to disable scheduling of ready tasks,
    # so that the private queue of ready tasks can be accessed via
    # scheduler._ready_queue
    pass


async def test_scheduler_task_no_dependency(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()

    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    task1 = DummyTask()
    task2 = DummyTask()
    await request_queue.put([task1, task2])

    ready1 = await scheduler._ready_queue.get()
    ready2 = await scheduler._ready_queue.get()
    assert ready1.tid == task1.tid
    assert ready2.tid == task2.tid

    assert scheduler._ready_queue.empty()

    await scheduler.cancel()


async def test_scheduler_task_bounded_queue(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue(
        maxsize=1
    )

    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    task1 = DummyTask()
    task2 = DummyTask()
    await request_queue.put([task1])
    await request_queue.put([task2])

    ready1 = await scheduler._ready_queue.get()
    ready2 = await scheduler._ready_queue.get()
    assert ready1.tid == task1.tid
    assert ready2.tid == task2.tid

    assert scheduler._ready_queue.empty()

    await scheduler.cancel()


async def test_scheduler_task_add_via_generator(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()

    task1: DummyTask | None = None
    task2: DummyTask | None = None

    def task_generator() -> Iterable[TaskBase]:
        nonlocal task1
        task1 = DummyTask()
        yield task1
        nonlocal task2
        task2 = DummyTask()
        yield task2

    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    await request_queue.put(task for task in task_generator())

    ready1 = await scheduler._ready_queue.get()
    ready2 = await scheduler._ready_queue.get()
    assert task1 is not None
    assert ready1.tid == task1.tid
    assert task2 is not None
    assert ready2.tid == task2.tid

    assert scheduler._ready_queue.empty()

    await scheduler.cancel()


async def test_scheduler_task_single_dependency(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()

    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    task1 = DummyTask()
    task2 = DummyTask(dependencies=[task1.tid])
    await request_queue.put([task1, task2])

    ready1 = await scheduler._ready_queue.get()
    assert ready1.tid == task1.tid
    assert scheduler._ready_queue.empty()

    await scheduler._task_events_queue.put(TaskEvent(source=task1.tid, state="done"))
    ready2 = await scheduler._ready_queue.get()
    assert ready2.tid == task2.tid
    assert scheduler._ready_queue.empty()

    await scheduler.cancel()


async def test_scheduler_task_double_dependency(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()

    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    task1 = DummyTask()
    task2 = DummyTask()
    task3 = DummyTask(dependencies=[task1.tid, task2.tid])

    await request_queue.put([task1, task2, task3])

    ready1 = await scheduler._ready_queue.get()
    assert ready1.tid == task1.tid
    assert not scheduler._ready_queue.empty()

    await scheduler._task_events_queue.put(TaskEvent(source=task1.tid, state="done"))
    assert not scheduler._ready_queue.empty()

    ready2 = await scheduler._ready_queue.get()
    assert ready2.tid == task2.tid
    assert scheduler._ready_queue.empty()

    await scheduler._task_events_queue.put(TaskEvent(source=task2.tid, state="done"))
    ready3 = await scheduler._ready_queue.get()
    assert ready3.tid == task3.tid
    assert scheduler._ready_queue.empty()

    await scheduler.cancel()


async def test_scheduler_task_multiple_dependents(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()

    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    task1 = DummyTask()
    task2 = DummyTask(dependencies=[task1.tid])
    task3 = DummyTask(dependencies=[task1.tid])
    task4 = DummyTask()

    await request_queue.put([task1, task2, task3, task4])

    ready1 = await scheduler._ready_queue.get()
    ready4 = await scheduler._ready_queue.get()
    assert {ready1.tid, ready4.tid} == {task1.tid, task4.tid}

    await scheduler._task_events_queue.put(TaskEvent(source=task1.tid, state="done"))

    ready2 = await scheduler._ready_queue.get()
    ready3 = await scheduler._ready_queue.get()
    assert {ready2.tid, ready3.tid} == {task2.tid, task3.tid}

    await scheduler.cancel()


async def test_scheduler_task_cancel_clears_state(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()

    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    task1 = DummyTask()
    await request_queue.put(task1)

    ready = await scheduler._ready_queue.get()
    assert ready.tid == task1.tid

    await scheduler.cancel()

    assert scheduler._requests == {}
    assert scheduler._dependents == {}
    assert scheduler._dependency_counts == {}


async def test_scheduler_task_missing_dependency() -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    scheduler = Scheduler(request_queue, event_queue=event_queue)
    scheduler.schedule()

    task1 = DummyTask()
    dummy_tid = uuid.uuid4()
    task2 = DummyTask(dependencies=[dummy_tid])

    await request_queue.put([task1, task2])

    event = await event_queue.get()
    assert isinstance(event, ErrorEvent)
    assert event.error is not None
    msg = f"Task dependency {dummy_tid} not found or invalid for task {task2.tid}."
    with pytest.raises(RuntimeError, match=msg):
        raise event.error

    await scheduler.cancel()


async def test_scheduler_task_event_with_error(monkeypatch: Any) -> None:
    request_queue: asyncio.Queue[TaskBase | Iterable[TaskBase]] = asyncio.Queue()
    scheduler = Scheduler(request_queue)
    monkeypatch.setattr(scheduler, "_submit_ready_tasks", _do_not_submit)
    scheduler.schedule()

    task1 = DummyTask()
    task2 = DummyTask()
    task3 = DummyTask(dependencies=[task1.tid, task2.tid])
    await request_queue.put([task1, task2, task3])
    await asyncio.sleep(0.0)

    assert task1.tid in scheduler._requests
    assert task2.tid in scheduler._requests
    assert task3.tid in scheduler._requests

    await scheduler._task_events_queue.put(TaskEvent(source=task1.tid, state="done"))
    await asyncio.sleep(0.0)

    assert task1.tid not in scheduler._requests
    assert task2.tid in scheduler._requests
    assert task3.tid in scheduler._requests

    await scheduler._task_events_queue.put(
        TaskEvent(source=task2.tid, state="done", error=RuntimeError("fail"))
    )
    await asyncio.sleep(0.0)

    assert task1.tid not in scheduler._requests
    assert task2.tid not in scheduler._requests
    assert task3.tid not in scheduler._requests

    await scheduler.cancel()


async def test_scheduler_poller() -> None:
    class MyTask(TaskBase):
        def __init__(self) -> None:
            super().__init__()

        async def run(self) -> None:  # noqa: PLR6301
            await asyncio.sleep(0.0)

    class MyPoller(PollerTask):
        def __init__(
            self,
            *,
            event_queue: asyncio.Queue[Event] | None = None,
            poll_interval: float = 0.01,
        ) -> None:
            super().__init__(event_queue=event_queue, poll_interval=poll_interval)

        def poll(self, tasks: set[TaskBase]) -> None:  # noqa: PLR6301
            for task in tasks:
                assert task.state == "running"
                task.mark_finished()

    poller = MyPoller().schedule()

    queue: asyncio.Queue[Event] = asyncio.Queue()

    task = MyTask()
    assert task.state == "pending"
    task.schedule(queue)

    state = await queue.get()
    assert isinstance(state, TaskEvent)
    assert state.state == "running"

    poller.add(task)

    state = await queue.get()
    assert isinstance(state, TaskEvent)
    assert state.state == "done"


async def test_scheduler_poller_cancel() -> None:
    release = False

    class MyTask(TaskBase):
        def __init__(self) -> None:
            super().__init__()

        async def run(self) -> None:  # noqa: PLR6301
            await asyncio.sleep(0.0)

    class MyPoller(PollerTask):
        def __init__(
            self,
            *,
            event_queue: asyncio.Queue[Event] | None = None,
            poll_interval: float = 0.01,
        ) -> None:
            super().__init__(event_queue=event_queue, poll_interval=poll_interval)

        def poll(self, tasks: set[TaskBase]) -> None:  # noqa: PLR6301
            for task in tasks:
                assert task.state == "running"
                if release:
                    task.mark_finished()

    poller = MyPoller().schedule()

    queue: asyncio.Queue[Event] = asyncio.Queue()

    task = MyTask()
    assert task.state == "pending"
    task.schedule(queue)

    state = await queue.get()
    assert isinstance(state, TaskEvent)
    assert state.state == "running"

    poller.add(task)

    await poller.cancel()

    release = True
    assert task.state == "running"  # type: ignore[comparison-overlap]

    task.mark_finished()

    state = await queue.get()
    assert isinstance(state, TaskEvent)
    assert state.state == "done"


class MainLoop(MainEventLoop):
    """Derived main loop for testing all task types."""

    def __init__(
        self,
        max_tasks: int | None = None,
        poller: PollerTask | None = None,
    ) -> None:
        super().__init__(max_tasks=max_tasks, poller=poller)
        self.tasks: dict[uuid.UUID, FastTask] = {}
        self.remaining: set[uuid.UUID] = set()

    async def add_tasks(self, tasks: TaskBase | Iterable[TaskBase]) -> None:
        await super().add_tasks(tasks)
        for task in tasks if isinstance(tasks, Iterable) else [tasks]:
            assert isinstance(task, FastTask)
            self.tasks[task.tid] = task
            self.remaining.add(task.tid)

    async def handle_event(self, event: Event) -> None:
        if isinstance(event, TaskEvent) and event.state == "done":
            assert isinstance(event.source, uuid.UUID)
            self.remaining.discard(event.source)
            if not self.remaining:
                self.stop()


class FastTask(TaskBase):
    """Fast AsyncTask for testing dependencies."""

    def __init__(
        self,
        *,
        dependencies: list[uuid.UUID] | None = None,
        main_loop: MainLoop | None = None,
        is_background_task: bool = False,
        block: asyncio.Event | None = None,
        fail: bool = False,
    ) -> None:
        super().__init__(dependencies=dependencies)
        self.main_loop = main_loop
        self._is_background_task = is_background_task
        self._block = block
        self._fail = fail
        self.finished: bool = False
        self.deps_satisfied: bool = False

    async def run(self) -> None:
        if self._fail:
            raise RuntimeError("fail")  # noqa: EM101
        if self._block is not None:
            await self._block.wait()
        if self.main_loop:
            self.deps_satisfied = all(
                self.main_loop.tasks[tid].finished for tid in self.dependencies
            )
        self.finished = True
        await asyncio.sleep(0.0)
        if not self._is_background_task:
            self.mark_finished()


class Poller(PollerTask):
    def __init__(
        self,
        *,
        event_queue: asyncio.Queue[Event] | None = None,
        poll_interval: float = 0.01,
    ) -> None:
        super().__init__(event_queue=event_queue, poll_interval=poll_interval)

    def poll(self, tasks: set[TaskBase]) -> None:  # noqa: PLR6301
        for task in tasks:
            if task.state == "running":
                task.mark_finished()


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_single_task_completion(is_background_task: bool) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)
    task = FastTask(is_background_task=is_background_task)
    await loop.add_tasks([task])
    await loop.run()
    assert task.state == "done"


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_double_task_completion(is_background_task: bool) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)
    task1 = FastTask(is_background_task=is_background_task)
    task2 = FastTask(is_background_task=is_background_task)
    await loop.add_tasks([task1, task2])
    await loop.run()
    assert task1.state == "done"
    assert task2.state == "done"


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_dependency_enforced(is_background_task: bool) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)

    task1 = FastTask(is_background_task=is_background_task, main_loop=loop)
    task2 = FastTask(
        is_background_task=is_background_task, dependencies=[task1.tid], main_loop=loop
    )

    await loop.add_tasks([task1, task2])
    await loop.run()

    assert task1.state == "done"
    assert task2.state == "done"
    assert task2.deps_satisfied


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_scheduler_main_multiple_dependents(is_background_task: bool) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)
    task1 = FastTask(is_background_task=is_background_task)
    task2 = FastTask(
        is_background_task=is_background_task, dependencies=[task1.tid], main_loop=loop
    )
    task3 = FastTask(
        is_background_task=is_background_task, dependencies=[task1.tid], main_loop=loop
    )
    await loop.add_tasks([task1, task2, task3])
    await loop.run()

    assert task1.state == "done"
    assert task2.state == "done"
    assert task3.state == "done"
    assert task2.deps_satisfied
    assert task3.deps_satisfied


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_scheduler_main_multiple_dependencies(is_background_task: bool) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)
    task1 = FastTask(is_background_task=is_background_task)
    task2 = FastTask(is_background_task=is_background_task)
    task3 = FastTask(
        is_background_task=is_background_task,
        dependencies=[task1.tid, task2.tid],
        main_loop=loop,
    )
    await loop.add_tasks([task1, task2, task3])
    await loop.run()

    assert task1.state == "done"
    assert task2.state == "done"
    assert task3.state == "done"
    assert task3.deps_satisfied


@pytest.mark.parametrize("is_background_task", [True, False])
@pytest.mark.parametrize(
    ("max_tasks", "expected"),
    [
        (1, ["running", "done", "running", "done"]),
        (2, ["running", "running", "done", "done"]),
    ],
)
async def test_scheduler_main_semaphore_respected(
    is_background_task: bool,
    max_tasks: int,
    expected: list[str],
    monkeypatch: Any,
) -> None:
    poller = Poller()
    loop = MainLoop(max_tasks=max_tasks, poller=poller if is_background_task else None)

    task1 = FastTask(is_background_task=is_background_task)
    task2 = FastTask(is_background_task=is_background_task)

    await loop.add_tasks([task1, task2])

    events: list[Event] = []
    old_handle_event = loop.handle_event

    async def capture_event(event: Event) -> None:
        events.append(event)
        await old_handle_event(event)

    monkeypatch.setattr(loop, "handle_event", capture_event)

    await loop.run()

    for event, state in zip(events, expected, strict=True):
        assert isinstance(event, TaskEvent)
        assert event.state == state


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_scheduler_main_task_errors_raised(is_background_task: bool) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)
    task = FastTask(is_background_task=is_background_task, fail=True)
    await loop.add_tasks([task])
    with pytest.raises(RuntimeError, match="fail"):
        await loop.run()


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_scheduler_main_task_error_events(
    is_background_task: bool, monkeypatch: Any
) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)
    task = FastTask(is_background_task=is_background_task, fail=True)
    await loop.add_tasks([task])

    events: list[Event] = []

    async def capture_event(event: Event) -> None:  # noqa: RUF029
        events.append(event)
        assert isinstance(event.source, uuid.UUID)
        loop.remaining.discard(event.source)
        if not loop.remaining:
            loop.stop()

    monkeypatch.setattr(loop, "handle_error", capture_event)
    await loop.run()

    assert isinstance(events[-1], TaskEvent)
    assert isinstance(events[-1].error, RuntimeError)
    assert str(events[-1].error) == "fail"


@pytest.mark.parametrize("is_background_task", [True, False])
async def test_scheduler_main_loop_stop(
    is_background_task: bool, monkeypatch: Any
) -> None:
    poller = Poller()
    loop = MainLoop(poller=poller if is_background_task else None)

    block = asyncio.Event()
    task1 = FastTask(is_background_task=is_background_task, block=block)
    task2 = FastTask(is_background_task=is_background_task, block=block)
    task3 = FastTask(
        is_background_task=is_background_task, block=block, dependencies=[task1.tid]
    )

    await loop.add_tasks([task1, task2, task3])

    events: list[Event] = []

    async def capture_event(event: Event) -> None:  # noqa: RUF029
        events.append(event)
        if len(events) == 2:
            loop.stop()

    monkeypatch.setattr(loop, "handle_event", capture_event)

    await loop.run()

    for event, state in zip(events, ["running", "running"], strict=True):
        assert isinstance(event, TaskEvent)
        assert event.state == state
    assert task1.state == "cancelled"
    assert task2.state == "cancelled"
    assert task3.state == "pending"
