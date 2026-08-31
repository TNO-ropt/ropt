"""Tests that submit to a real, already-installed HPC cluster.

Run with `--hpc=QUEUE`, naming a queue of the `pysqa` configuration installed
alongside ropt. `--tmp=DIR` says where job files go; it must be on a filesystem
the compute nodes share, must not already exist, and is removed afterwards:

    pytest tests/test_hpc_cluster.py --hpc=normal --tmp=/scratch/me/ropt

These tests exist for the things a mocked `pysqa` cannot show: that the
scheduler answers with a `jobid` column, that it reports several jobs at once,
that a cancelled job really leaves the queue, that the submitting interpreter
resolves on a compute node, and that a job's captured output survives to explain
a failure. ropt's own arithmetic — `retries`, `query_retries`, worker limits —
stays in the mocked tests, where it is deterministic and fast.

The submitted callables are all resolvable from the standard library, because a
compute node runs `sys.executable -m ropt.components.executors` and cannot
import this test module.
"""

from __future__ import annotations

import asyncio
import operator
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ropt.components.executors import HPCExecutor, Submission, WorkItem
from ropt.exceptions import ExecutorFailure

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.hpc, pytest.mark.asyncio, pytest.mark.timeout(600)]

_SLEEP_SECONDS = 60

_HOLD_SECONDS = 8

_CONCURRENT_JOBS = 4

_FAILING_JOB = "import os, sys; print('ropt-marker'); sys.stdout.flush(); os._exit(1)"


def _holding_job(value: int) -> str:
    # Held open long enough to be observed in the queue, and returning a value
    # that identifies which work item it came from.
    return f"__import__('time').sleep({_HOLD_SECONDS}) or {value}"


@pytest.fixture(name="hpc_queue", scope="session")
def hpc_queue_fixture(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--hpc"))


@pytest.fixture(name="hpc_root", scope="session")
def hpc_root_fixture(pytestconfig: pytest.Config) -> Iterator[Path]:
    root = Path(str(pytestconfig.getoption("--tmp"))).resolve()
    if root.exists():
        msg = f"--tmp must not exist, it is removed afterwards: {root}"
        raise FileExistsError(msg)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(name="workdir")
def workdir_fixture(hpc_root: Path, request: pytest.FixtureRequest) -> Path:
    workdir = hpc_root / str(request.node.name)
    workdir.mkdir()
    return workdir


@pytest.fixture(name="executor")
def executor_fixture(workdir: Path, hpc_queue: str) -> Iterator[Any]:
    executors: list[HPCExecutor] = []

    def _make(**kwargs: Any) -> HPCExecutor:
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("interval", 2)
        kwargs.setdefault("cores", 1)
        executor = HPCExecutor(workdir=workdir, queue=hpc_queue, **kwargs)
        executors.append(executor)
        return executor

    try:
        yield _make
    finally:
        for executor in executors:
            executor.cancel()


async def test_hpc_cluster_resolves_the_requested_queue(  # ruff: ignore[unused-async]
    executor: Any,
) -> None:
    adapter = executor()._queue_adapter  # ruff: ignore[private-member-access]
    assert adapter.queue_list is not None


async def test_hpc_cluster_submitted_job_returns_its_result(
    executor: Any,
) -> None:
    submission = Submission([WorkItem(function=operator.add, args=(40, 2))])
    hpc = executor()
    async with asyncio.TaskGroup() as tg:
        await hpc.start(tg)
        hpc.submit(submission)
        collected: list[Any] = []
        await asyncio.to_thread(
            submission.collect, lambda item: collected.append(item.result)
        )
        hpc.cancel()
    assert collected == [42]


async def test_hpc_cluster_cancelled_job_disappears_from_the_scheduler(
    executor: Any,
) -> None:
    submission = Submission([WorkItem(function=time.sleep, args=(_SLEEP_SECONDS,))])
    hpc = executor()
    async with asyncio.TaskGroup() as tg:
        await hpc.start(tg)
        hpc.submit(submission)
        live = await asyncio.to_thread(_wait_for_live_jobs, hpc)
        assert live, "the scheduler reported no job for a submission still running"
        assert all(isinstance(job_id, int) for job_id in live)
        hpc.cancel()
    gone = await asyncio.to_thread(_wait_until_gone, hpc, live)
    assert gone, f"cancelled jobs are still queued: {live}"


async def test_hpc_cluster_concurrent_jobs_each_return_their_own_result(
    executor: Any,
) -> None:
    expected = {f"item-{index}": index * 10 for index in range(1, _CONCURRENT_JOBS + 1)}
    submission = Submission(
        [
            WorkItem(
                function=eval,  # ruff: ignore[suspicious-eval-usage]
                args=(_holding_job(value),),
                name=name,
            )
            for name, value in expected.items()
        ]
    )
    hpc = executor(workers=_CONCURRENT_JOBS)
    collected: dict[str | None, Any] = {}
    async with asyncio.TaskGroup() as tg:
        await hpc.start(tg)
        hpc.submit(submission)
        peak, _ = await asyncio.gather(
            asyncio.to_thread(_peak_live_jobs, hpc),
            asyncio.to_thread(
                submission.collect,
                lambda item: collected.update({item.name: item.result}),
            ),
        )
        hpc.cancel()
    assert collected == expected
    # Pending jobs are queued too, so this holds on a cluster too busy to run
    # them side by side; only a scheduler reporting one job at a time fails it.
    assert peak > 1, (
        f"the scheduler never reported more than one job at once (peak: {peak}), "
        "so nothing here exercised concurrent submission"
    )


async def test_hpc_cluster_failed_job_reports_where_its_output_is(
    executor: Any, workdir: Path
) -> None:
    submission = Submission([WorkItem(function=exec, args=(_FAILING_JOB,))])
    hpc = executor(retries=3)
    async with asyncio.TaskGroup() as tg:
        await hpc.start(tg)
        hpc.submit(submission)
        collected: list[Any] = []
        await asyncio.to_thread(
            submission.collect, lambda item: collected.append(item.result)
        )
        hpc.cancel()
    assert len(collected) == 1
    failure = collected[0]
    assert isinstance(failure, ExecutorFailure), failure
    assert str(workdir) in str(failure), (
        "The failure does not say where the job's output is, which is all a job "
        f"that dies leaves behind.\nfailure: {failure}"
    )
    # Whether the tail reached the message depends on when a shared filesystem
    # shows it; that the file exists at all depends on the installed submission
    # script, and does not.
    assert _listing(workdir, "*.txt"), (
        "The job wrote no output file, so nothing can explain a failed job. The "
        "installed submission script must direct the job's output there with "
        "'#SBATCH --output={{output}}' (pysqa's own example template hardcodes "
        f"a name instead, and will not do).\nworkdir holds: {_listing(workdir)}"
    )


def _listing(workdir: Path, pattern: str = "*") -> list[str]:
    return sorted(path.name for path in workdir.glob(pattern))


def _wait_for_live_jobs(hpc: HPCExecutor, timeout: float = 120.0) -> set[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = hpc._live_job_ids()  # ruff: ignore[private-member-access]
        if live:
            return live
        time.sleep(2)
    return set()


def _peak_live_jobs(hpc: HPCExecutor, timeout: float = 300.0) -> int:
    deadline = time.monotonic() + timeout
    peak = 0
    seen = False
    while time.monotonic() < deadline:
        live = hpc._live_job_ids()  # ruff: ignore[private-member-access]
        peak = max(peak, len(live))
        if live:
            seen = True
        elif seen:
            break
        time.sleep(0.5)
    return peak


def _wait_until_gone(hpc: HPCExecutor, jobs: set[int], timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not jobs & hpc._live_job_ids():  # ruff: ignore[private-member-access]
            return True
        time.sleep(2)
    return False
