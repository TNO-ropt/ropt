"""The abort scope of one optimization, evaluation or `run` call.

Aborts travel down and failures travel up. The scope of a call is nested in
the current scope where the call is made, and in the scope of the session of
the pool it is given. A call that fails aborts the work it started, and its
exception is raised to its caller; the calls beside it are not aborted.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ropt._components.concurrency import AbortScope, current_abort_scope
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError, ExecutorStopped

if TYPE_CHECKING:
    from collections.abc import Generator

    from ._session import Session


@contextlib.contextmanager
def linked_abort_scope(session: Session | None) -> Generator[AbortScope]:
    root = None if session is None else session._open_abort_scope()  # ruff: ignore[private-member-access]
    scope = AbortScope()
    with scope.nested_in(current_abort_scope()), scope.nested_in(root):
        try:
            yield scope
        except ExecutorStopped:
            # Not a failure of this call, so it is reported the way an abort is
            # rather than escaping.
            raise AbortedError(ExitCode.EXECUTOR_SHUT_DOWN) from None
        except Exception:
            # `scope.aborting` means this call was aborted rather than failing.
            if not scope.aborting:
                scope.abort(ExitCode.ABORTED_ON_ERROR)
            raise
        except BaseException:
            # Not a failure, but the work started from this call must not
            # outlive it.
            scope.abort(ExitCode.ABORTED)
            raise
