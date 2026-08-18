"""Names for the tasks of a single run.

Only the `HPCExecutor` uses task names, where they are also the task id and the
filename base, so they must be unique within the executor. The run id that makes
them unique comes from the session, which is the only tie to it.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.components.evaluators import EvaluationFunctionContext, NameCallback
    from ropt.components.executors import Executor

    from ._session import _Session


def _name_task(run_id: int, contexts: Sequence[EvaluationFunctionContext]) -> str:
    context = contexts[0]
    name = f"run{run_id}-b{context.batch_id}-r{context.realization}"
    if context.perturbation >= 0:
        name = f"{name}-p{context.perturbation}"
    return name


def make_task_namer(
    session: _Session | None, executor: Executor | None
) -> NameCallback | None:
    """Build an auto-naming callback for a single run's tasks.

    Names have the form `run{id}-b{batch}-r{realization}[-p{perturbation}]`,
    where `id` is unique within the executor and the `-p` suffix is dropped for
    unperturbed evaluations. Only the `HPCExecutor` uses these names; for other
    executors the callback is harmless.

    Args:
        session:  The active session, or `None` on the sequential floor.
        executor: The active executor, or `None` on the sequential floor.

    Returns:
        A naming callback, or `None` when there is no executor.
    """
    if session is None or executor is None:
        return None
    return partial(_name_task, session.next_run_id())
