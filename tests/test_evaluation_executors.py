from __future__ import annotations

import asyncio
import sys
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.evaluators import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    ParallelEvaluator,
)
from ropt.components.evaluators._parallel_evaluator import _abort, _handle_result
from ropt.components.event_handlers import ResultsHandler
from ropt.components.executors import (
    HPCExecutor,
    MultiprocessingExecutor,
    ResultsQueue,
    Task,
    ThreadingExecutor,
)
from ropt.components.executors._multiprocessing_executor import (
    _HAVE_CLOUDPICKLE,
    _run_cloudpickled,
)
from ropt.context import EnOptContext
from ropt.evaluation import EvaluationBatchContext
from ropt.exceptions import ExecutorFailure, ExecutorStopped, TransferError
from ropt.workflow._basic_optimizer import BasicOptimizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from ropt.components.executors import Executor
    from ropt.results import FunctionResults

try:
    import cloudpickle
    import pandas as pd
    import pysqa  # ruff: ignore[unused-import]

    from ropt.components.executors.__main__ import run_task

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(5)]


class _ResultProcessor:
    def __init__(self) -> None:
        self.results: set[int] = set()

    def process_results(
        self,
        result_queue: ResultsQueue,
        count: int,
        finished_event: asyncio.Event,
    ) -> None:
        for _ in range(count):
            task = result_queue.get()
            if task is None:
                break
            assert isinstance(task, Task)
            assert task.result is not None
            self.results.add(task.result)
        finished_event.set()


def _function(input_value: int, *, raise_error: bool = False) -> int:
    if raise_error:
        msg = f"Test error in function {input_value}"
        raise ValueError(msg)
    return input_value + 1


def _raise_unpicklable_error(_input: int) -> int:
    raise ValueError(threading.Lock())  # a lock cannot be (cloud)pickled


def _construct_handler_in_worker(_: int) -> str:
    return type(ResultsHandler()).__name__


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_ok(
    executor_name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    result_queue: ResultsQueue = ResultsQueue()
    tasks = [
        Task(function=_function, args=(idx,), results_queue=result_queue)
        for idx in range(2)
    ]
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: Executor = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor.is_running()
    all_processed = asyncio.Event()
    result_processor = _ResultProcessor()
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        tg.create_task(
            asyncio.to_thread(
                result_processor.process_results,
                result_queue,
                len(tasks),
                all_processed,
            )
        )
        assert executor.is_running()
        for task in tasks:
            await executor.task_queue.put(task)
        await all_processed.wait()
        executor.cancel()
    assert result_processor.results == {1, 2}
    assert not executor.is_running()


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_executor_refuses_to_overwrite_existing_task_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    (tmp_path / "job1.out").touch()  # a stale file, e.g. from another executor
    task = Task(
        function=_function, args=(0,), results_queue=ResultsQueue(), name="job1"
    )
    await asyncio.sleep(0)  # this module runs tests on the event loop
    with pytest.raises(RuntimeError, match="already exist"):
        executor._submit(task)  # ruff: ignore[private-member-access]


@pytest.mark.skipif(not _HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_cloudpickled_worker_records_the_worker_traceback_as_a_note() -> None:
    payload = cloudpickle.dumps((partial(_function, 0, raise_error=True), (), {}))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_cloudpickled(payload)
    assert not ok
    exc = cloudpickle.loads(blob)
    assert isinstance(exc, ValueError)
    assert any("Traceback" in note for note in exc.__notes__)


@pytest.mark.skipif(not _HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_cloudpickled_worker_wraps_an_unpicklable_exception() -> None:
    payload = cloudpickle.dumps((_raise_unpicklable_error, (0,), {}))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_cloudpickled(payload)
    assert not ok
    exc = cloudpickle.loads(blob)
    assert isinstance(exc, RuntimeError)
    assert any("Traceback" in note for note in exc.__notes__)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_worker_may_construct_workflow_objects() -> None:
    result_queue: ResultsQueue = ResultsQueue()
    task = Task(
        function=_construct_handler_in_worker, args=(0,), results_queue=result_queue
    )
    executor = MultiprocessingExecutor(workers=1)
    results: list[Any] = []
    finished = asyncio.Event()

    def _collect() -> None:
        item = result_queue.get()
        assert isinstance(item, Task)
        results.append(item.result)
        finished.set()

    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        tg.create_task(asyncio.to_thread(_collect))
        await executor.task_queue.put(task)
        await finished.wait()
        executor.cancel()
    assert results == ["ResultsHandler"]


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_error(
    executor_name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    result_queue: ResultsQueue = ResultsQueue()
    tasks = [
        Task(
            function=_function,
            args=(idx,),
            kwargs={"raise_error": True},
            results_queue=result_queue,
        )
        for idx in range(2)
    ]
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: Executor = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor.is_running()
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor.is_running()
        for task in tasks:
            await executor.task_queue.put(task)
        # A user-code exception is delivered on the results queue as the
        # original exception, and does not tear the executor down.
        item = await asyncio.to_thread(result_queue.get, timeout=10)
        assert isinstance(item, ValueError)
        assert "Test error in function" in str(item)
        assert executor.is_running()
        expects_notes = executor_name == "hpc" or (
            executor_name == "multiprocessing" and _HAVE_CLOUDPICKLE
        )
        if expects_notes:
            notes = getattr(item, "__notes__", [])
            assert any("Test error in function" in note for note in notes)
            assert any("Traceback" in note for note in notes)
        executor.cancel()
    assert not executor.is_running()


initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 8,
        },
        "backend": {
            "convergence_tolerance": 1e-2,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.001,
        },
        "gradient": {
            "number_of_perturbations": 3,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def _opt_function(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    test_functions: Any,
    *,
    raise_error: bool = False,
) -> EvaluationFunctionResult:
    if raise_error:
        msg = "Test error in function"
        raise ValueError(msg)
    return EvaluationFunctionResult(
        objectives=np.fromiter(
            (func(variables, context) for func in test_functions), dtype=np.float64
        )
    )


def _opt_workflow(
    executor: Executor,
    config: dict[str, Any],
    test_function: EvaluationFunctionCallback,
) -> FunctionResults | None:
    evaluator = ParallelEvaluator(function=test_function, executor=executor)
    optimizer = BasicOptimizer(config=config, evaluator=evaluator)
    optimizer.run(initial_values)
    return optimizer.results


if _TEST_HPC:

    class MockedHPCAdapter:
        def __init__(self, path: Path) -> None:
            self._path = path
            self._jobs: dict[int, str] = {}
            self._job_id = 0

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            *_, input_file, output_file = command.split()
            threading.Thread(
                target=run_task, args=(input_file, output_file), daemon=True
            ).start()
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

        def get_status_of_my_jobs(self) -> pd.DataFrame:
            running = [
                job_id
                for job_id, job_name in self._jobs.items()
                if not (self._path / f"{job_name}.out").exists()
            ]
            self._jobs = {job_id: self._jobs[job_id] for job_id in running}
            return pd.DataFrame(list(self._jobs.keys()), columns=["jobid"])


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_evaluator_ok(
    config: dict[str, Any],
    objective: Any,
    executor_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: Executor = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor.is_running()
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor.is_running()
        results = await asyncio.to_thread(
            _opt_workflow,
            executor,
            config,
            objective(),
        )
        executor.cancel()
    assert not executor.is_running()

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_evaluator_error(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    executor_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: Executor = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor.is_running()
    with pytest.raises(ExceptionGroup) as excinfo:  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            assert executor.is_running()
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(_opt_function, test_functions=test_functions, raise_error=True),
            )
            executor.cancel()
    # The user-code error surfaces as the original exception (not an exit code),
    # wrapped by the consumer's task group when it leaves the block.
    matched, _ = excinfo.value.split(ValueError)
    assert matched is not None
    assert all("Test error in function" in str(err) for err in matched.exceptions)
    assert not executor.is_running()


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
    ],
)
async def test_executor_survives_user_code_error_and_is_reusable(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    objective: Any,
    executor_name: str,
) -> None:
    match executor_name:
        case "threading":
            executor: Executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        # A user-code error aborts only its own evaluation; the executor keeps
        # running, so a caught error does not prevent a subsequent reuse.
        with pytest.raises(ValueError, match="Test error in function"):
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(_opt_function, test_functions=test_functions, raise_error=True),
            )
        assert executor.is_running()
        results = await asyncio.to_thread(_opt_workflow, executor, config, objective())
        assert results is not None
        assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
        executor.cancel()
    assert not executor.is_running()


async def test_error_escaping_the_body_closes_the_executor(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
) -> None:
    # No explicit executor.cancel(): an error escaping the block must still
    # close the executor through the task group's teardown.
    executor = ThreadingExecutor(workers=2)
    with pytest.raises(ExceptionGroup):  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(_opt_function, test_functions=test_functions, raise_error=True),
            )
    assert not executor.is_running()


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_evaluator_two_optimizations(
    config: dict[str, Any],
    objective: Any,
    executor_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: Executor = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor.is_running()
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor.is_running()
        results_list = await asyncio.gather(
            *(
                asyncio.to_thread(
                    _opt_workflow,
                    executor,
                    config,
                    objective(),
                )
                for _ in range(2)
            )
        )
        executor.cancel()
    assert not executor.is_running()

    assert len(results_list) == 2
    for results in results_list:
        assert results is not None
        assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize("bundle_size", [1, 2, 4, 0])
@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
    ],
)
async def test_groups_tasks(
    config: dict[str, Any],
    objective: Any,
    executor_name: str,
    bundle_size: int,
) -> None:
    match executor_name:
        case "threading":
            executor: Executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)

    task_sizes: list[int] = []
    original_put = executor.task_queue.put

    async def _counting_put(task: Task) -> None:
        task_sizes.append(len(task.args[1]))
        await original_put(task)

    executor.task_queue.put = _counting_put  # type: ignore[assignment]

    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        evaluator = ParallelEvaluator(
            function=objective(),
            executor=executor,
            bundle_size=bundle_size,
        )
        optimizer = BasicOptimizer(config=config, evaluator=evaluator)
        await asyncio.to_thread(optimizer.run, initial_values)
        executor.cancel()

    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )
    assert task_sizes, "No tasks were submitted"
    expected_max = max(task_sizes) if bundle_size == 0 else bundle_size
    for size in task_sizes:
        assert 1 <= size <= expected_max


async def test_invalid_bundle_size() -> None:  # ruff: ignore[unused-async]
    executor = ThreadingExecutor(workers=1)
    with pytest.raises(ValueError, match="bundle_size"):
        ParallelEvaluator(
            function=lambda variables, context: EvaluationFunctionResult(  # ruff: ignore[unused-lambda-argument]
                objectives=0.0
            ),
            executor=executor,
            bundle_size=-1,
        )


async def test_task_put_error_delivers_exception_and_closes_queue() -> None:  # ruff: ignore[unused-async]
    result_queue = ResultsQueue()
    task = Task(function=_function, args=(0,), results_queue=result_queue)
    error = ValueError("Test error in function")
    task.put_error(error)
    assert result_queue.get_nowait() is error
    assert result_queue.closed
    task.put_result(1)
    assert result_queue.empty()


async def test_abort_reraises_queued_exception() -> None:  # ruff: ignore[unused-async]
    result_queue = ResultsQueue()
    error = ValueError("Test error in function")
    result_queue.put(error)
    with pytest.raises(ValueError, match="Test error in function") as excinfo:
        _abort(result_queue)
    assert excinfo.value is error


async def test_abort_without_queued_exception_has_no_cause() -> None:  # ruff: ignore[unused-async]
    with pytest.raises(ExecutorStopped) as excinfo:
        _abort(ResultsQueue())
    assert excinfo.value.__cause__ is None


async def test_put_tasks_does_not_raise_executor_stopped_into_task_group(
    config: dict[str, Any], objective: Any
) -> None:
    # A stopped executor is reported to the caller by eval()'s consumer loop, so
    # the unawaited producer task must return cleanly instead of raising
    # EXECUTOR_STOPPED into the executor's task group, which would tear it down.
    executor = ThreadingExecutor(workers=1)
    assert not executor.is_running()
    evaluator = ParallelEvaluator(function=objective(), executor=executor)
    batch_context = EvaluationBatchContext(
        context=EnOptContext.model_validate(config),
        active=np.array([True]),
        realizations=np.array([0], dtype=np.intc),
    )
    await evaluator._put_tasks(  # ruff: ignore[private-member-access]
        np.zeros((1, 1)), batch_context, ResultsQueue(), 0
    )


class _FatalError(BaseException):
    pass


async def test_worker_base_exception_propagates_into_task_group() -> None:
    def _raise_fatal(input_value: int) -> int:  # ruff: ignore[unused-function-argument]
        msg = "fatal"
        raise _FatalError(msg)

    result_queue: ResultsQueue = ResultsQueue()
    executor = ThreadingExecutor(workers=1)
    with pytest.raises(BaseExceptionGroup) as excinfo:  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            await executor.task_queue.put(
                Task(function=_raise_fatal, args=(0,), results_queue=result_queue)
            )
    matched, _ = excinfo.value.split(_FatalError)
    assert matched is not None
    assert not executor.is_running()


async def test_handle_result_records_executor_failure_as_nan() -> None:  # ruff: ignore[unused-async]
    results = np.zeros((2, 1), dtype=np.float64)
    bundle = [
        (
            np.zeros(2, dtype=np.float64),
            EvaluationFunctionContext(
                realization=0, perturbation=-1, batch_id=0, eval_idx=idx
            ),
        )
        for idx in range(2)
    ]
    task = Task(
        function=_function,
        args=(None, bundle),
        results_queue=ResultsQueue(),
        result=ExecutorFailure("Background process was killed"),
    )
    handled = _handle_result(task, results, {}, objective_count=1, eval_count=2)
    assert handled == 2
    assert np.all(np.isnan(results))


@pytest.mark.slow
@pytest.mark.timeout(60)
async def test_multiprocessing_unguarded_main_reports_startup_error(
    tmp_path: Path,
) -> None:
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import asyncio\n\n"
        "from ropt.workflow import dispatch_tasks\n\n\n"
        "def work() -> int:\n"
        "    return 1\n\n\n"
        'asyncio.run(dispatch_tasks([work], executor="multiprocessing"))\n'
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode != 0
    assert b"Could not start MultiprocessingExecutor workers" in stderr


async def _run_multiprocessing_tasks(
    tasks: list[Task], result_queue: ResultsQueue, *, workers: int = 2
) -> list[Any]:
    collected: list[Any] = []
    finished = asyncio.Event()

    def collect() -> None:
        for _ in range(len(tasks)):
            item = result_queue.get()
            if item is None:
                break
            assert isinstance(item, Task)
            collected.append(item.result)
        finished.set()

    executor = MultiprocessingExecutor(workers=workers)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        tg.create_task(asyncio.to_thread(collect))
        for task in tasks:
            await executor.task_queue.put(task)
        await finished.wait()
        executor.cancel()
    return collected


@pytest.mark.slow
async def test_multiprocessing_cloudpickles_task_functions_and_results() -> None:
    def make_adder(offset: int) -> Callable[[int], int]:
        def add(value: int) -> int:
            return value + offset

        return add

    def local_double(value: int) -> int:
        return value * 2

    def make_callable(value: int) -> Callable[[], int]:
        return lambda: value

    result_queue: ResultsQueue = ResultsQueue()
    tasks = [
        Task(
            function=lambda value: value + 100,
            args=(1,),
            results_queue=result_queue,
        ),
        Task(function=make_adder(10), args=(2,), results_queue=result_queue),
        Task(function=local_double, args=(3,), results_queue=result_queue),
        Task(function=make_callable, args=(42,), results_queue=result_queue),
    ]
    results = await _run_multiprocessing_tasks(tasks, result_queue)
    assert sorted(value for value in results if isinstance(value, int)) == [6, 12, 101]
    returned = [value for value in results if callable(value)]
    assert len(returned) == 1
    assert returned[0]() == 42


@pytest.mark.slow
async def test_multiprocessing_unserializable_payload_reports_error() -> None:
    lock = threading.Lock()

    def use_lock() -> Any:
        return lock

    result_queue: ResultsQueue = ResultsQueue()
    task = Task(function=use_lock, results_queue=result_queue)
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        await executor.task_queue.put(task)
        # The serialization failure is delivered as the exception, not a teardown.
        item = await asyncio.to_thread(result_queue.get, timeout=30)
        assert isinstance(item, BaseException)
        assert "pickle" in str(item).lower()
        assert executor.is_running()
        executor.cancel()
    assert not executor.is_running()


@pytest.mark.slow
async def test_multiprocessing_without_cloudpickle_rejects_lambda(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor._HAVE_CLOUDPICKLE",
        False,
    )
    result_queue: ResultsQueue = ResultsQueue()
    task = Task(function=lambda: 1, results_queue=result_queue)
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        await executor.task_queue.put(task)
        item = await asyncio.to_thread(result_queue.get, timeout=30)
        assert isinstance(item, RuntimeError)
        assert "ropt[cloudpickle]" in str(item)
        assert executor.is_running()
        executor.cancel()
    assert not executor.is_running()


def _return_captured(_handler: Any) -> int:
    return 0


def _opt_function_capturing_handler(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    test_functions: Any,
    handler: Any,  # ruff: ignore[unused-function-argument]
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(
        objectives=np.fromiter(
            (func(variables, context) for func in test_functions), dtype=np.float64
        )
    )


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_task_capturing_a_workflow_object_raises_transfer_error() -> None:
    result_queue: ResultsQueue = ResultsQueue()
    task = Task(
        function=_return_captured,
        args=(ResultsHandler(),),
        results_queue=result_queue,
    )
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        await executor.task_queue.put(task)
        item = await asyncio.to_thread(result_queue.get, timeout=30)
        assert isinstance(item, TransferError)
        assert executor.is_running()
        executor.cancel()
    assert not executor.is_running()


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_transfer_error_from_parallel_evaluation_bubbles_up(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
) -> None:
    executor = MultiprocessingExecutor(workers=1)
    with pytest.raises(ExceptionGroup) as excinfo:  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(
                    _opt_function_capturing_handler,
                    test_functions=test_functions,
                    handler=ResultsHandler(),
                ),
            )
            executor.cancel()
    assert any(isinstance(err, TransferError) for err in excinfo.value.exceptions)


async def test_abort_reraises_transfer_error() -> None:  # ruff: ignore[unused-async]
    result_queue = ResultsQueue()
    error = TransferError("Workflow objects cannot be used in a worker process: X.")
    result_queue.put(error)
    with pytest.raises(TransferError):
        _abort(result_queue)
