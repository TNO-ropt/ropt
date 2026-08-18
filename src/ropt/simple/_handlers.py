"""Shared result handlers scoped by ``with ropt.handlers(...)`` blocks.

A ``handlers()`` block owns one `EventDispatcher` (started on the ambient
session's task group) holding exactly the handlers listed for that block. Runs
inside the block deliver their events to the innermost block's dispatcher, which
serializes the handlers across concurrent runs.

Blocks nest. By default a nested block *inherits* the enclosing blocks'
handlers: it steals them into its own dispatcher for the block's duration and
gives them back on exit, so those handlers also aggregate the nested runs. Pass
``inherit=False`` to skip that and re-list only the enclosing handlers you want.
Because an enclosing block is suspended while a nested block runs, a handler is
always attached to exactly one dispatcher at a time.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Self

from ropt.components.event_handlers import (
    EventDispatcher,
    EventForwardHandler,
    EventHandler,
)
from ropt.exceptions import WorkflowError

from ._report import make_report_handler
from ._session import _acquire_session, _release_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.components.compute_steps import ComputeStep
    from ropt.enums import EnOptEventType

    from ._report import ReportCallback
    from ._session import _Session


_handler_stack: ContextVar[tuple[HandlerScope, ...]] = ContextVar(
    "ropt_simple_handlers", default=()
)

_IN_USE = (
    "This handler is already in use and cannot join a handlers() block. A "
    "handler passed to optimize() as a local handler stays bound to its run, "
    "and one already held by another open block cannot be shared twice. Use a "
    "separate handler here."
)


class HandlerScope:
    """A block of shared result handlers, opened by `handlers`.

    Aggregates results across every run in the block. A compute step reaches the
    active block with `current_handlers` and feeds its events to the shared
    handlers with `attach_to`, which is the only method meant for callers
    outside this package.
    """

    def __init__(
        self,
        scope_handlers: Sequence[tuple[EventHandler, bool]],
        *,
        inherit: bool,
    ) -> None:
        self._scope_handlers = tuple(scope_handlers)
        self._inherit_handlers = inherit
        self._session: _Session | None = None
        self._session_token: Token[_Session | None] | None = None
        self._stack_token: Token[tuple[HandlerScope, ...]] | None = None
        self._event_dispatcher: EventDispatcher | None = None
        self._current_handlers: set[EventHandler] = set()
        self._migrated_handlers: list[tuple[EventHandler, HandlerScope, bool]] = []
        self._open = False

    def attach_to(self, step: ComputeStep) -> None:
        """Forward the events of a run's compute step to this block's handlers.

        A fresh forwarding handler is added per step (one handler cannot serve
        several steps), carrying only the event types the shared handlers want.

        Args:
            step: The compute step whose events feed the shared handlers.
        """
        event_types: set[EnOptEventType] = set()
        for handler in self._current_handlers:
            event_types |= handler.event_types
        if event_types:
            step.add_event_handler(
                EventForwardHandler(self._running_dispatcher, event_types=event_types)
            )

    @property
    def _running_dispatcher(self) -> EventDispatcher:
        assert self._event_dispatcher is not None
        return self._event_dispatcher

    def _owns(self, handler: EventHandler) -> bool:
        return handler in self._current_handlers

    def _owned(self) -> list[EventHandler]:
        return list(self._current_handlers)

    def _give_up_handler(self, handler: EventHandler) -> bool:
        in_thread = self._running_dispatcher.remove_event_handler(handler)
        self._current_handlers.discard(handler)
        return in_thread

    def _take_back_handler(self, handler: EventHandler, *, in_thread: bool) -> None:
        self._running_dispatcher.add_event_handler(handler, run_in_thread=in_thread)
        self._current_handlers.add(handler)

    def __enter__(self) -> Self:
        # Entering twice would overwrite the first block's contextvar token and
        # leave this scope on the stack for good, feeding every later run to a
        # cancelled dispatcher.
        if self._open:
            msg = "Handlers() block is already open and cannot be entered again."
            raise WorkflowError(msg)
        self._open = True
        try:
            self._open_block()
        except BaseException:
            self._open = False
            raise
        return self

    def _open_block(self) -> None:
        session, session_token = _acquire_session()
        dispatcher = EventDispatcher()
        stack = _handler_stack.get()
        added: set[EventHandler] = set()
        migrated: list[tuple[EventHandler, HandlerScope, bool]] = []
        try:
            self._attach(dispatcher, stack, added, migrated)
            session.open_dispatcher(dispatcher)
        except BaseException:
            try:
                self._return_handlers(dispatcher, added, migrated)
            finally:
                session.close_dispatcher(dispatcher)
                _release_session(session, session_token)
            raise
        self._session = session
        self._session_token = session_token
        self._event_dispatcher = dispatcher
        self._current_handlers = added
        self._migrated_handlers = migrated
        self._stack_token = _handler_stack.set((*stack, self))

    def _attach(
        self,
        dispatcher: EventDispatcher,
        stack: tuple[HandlerScope, ...],
        added: set[EventHandler],
        migrated: list[tuple[EventHandler, HandlerScope, bool]],
    ) -> None:
        for handler, run_in_thread in self._scope_handlers:
            source = self._find_owner(stack, handler)
            if source is not None:
                in_thread = source._give_up_handler(handler)  # ruff: ignore[private-member-access]
                migrated.append((handler, source, in_thread))
            try:
                dispatcher.add_event_handler(handler, run_in_thread=run_in_thread)
            except WorkflowError as exc:
                # The low-level refusal is phrased in terms of dispatchers and
                # compute steps, neither of which this API ever hands out.
                if handler in added:
                    raise
                raise WorkflowError(_IN_USE) from exc
            added.add(handler)
        if self._inherit_handlers:
            for source in stack:
                for handler in source._owned():  # ruff: ignore[private-member-access]
                    in_thread = source._give_up_handler(handler)  # ruff: ignore[private-member-access]
                    migrated.append((handler, source, in_thread))
                    dispatcher.add_event_handler(handler, run_in_thread=in_thread)
                    added.add(handler)

    @staticmethod
    def _find_owner(
        stack: tuple[HandlerScope, ...], handler: EventHandler
    ) -> HandlerScope | None:
        for scope in reversed(stack):
            if scope._owns(handler):  # ruff: ignore[private-member-access]
                return scope
        return None

    @staticmethod
    def _return_handlers(
        dispatcher: EventDispatcher,
        added: set[EventHandler],
        migrated: list[tuple[EventHandler, HandlerScope, bool]],
    ) -> None:
        for handler in added:
            dispatcher.remove_event_handler(handler)
        for handler, source, in_thread in migrated:
            source._take_back_handler(handler, in_thread=in_thread)  # ruff: ignore[private-member-access]

    def __exit__(self, *_exc: object) -> None:
        assert self._stack_token is not None
        assert self._session is not None
        _handler_stack.reset(self._stack_token)
        try:
            self._return_handlers(
                self._running_dispatcher,
                self._current_handlers,
                self._migrated_handlers,
            )
        finally:
            try:
                self._session.close_dispatcher(self._running_dispatcher)
                _release_session(self._session, self._session_token)
            finally:
                self._close_block()

    def _close_block(self) -> None:
        self._session = None
        self._session_token = None
        self._stack_token = None
        self._event_dispatcher = None
        self._current_handlers = set()
        self._migrated_handlers = []
        self._open = False


def handlers(
    *handler: EventHandler,
    threaded: EventHandler | Sequence[EventHandler] = (),
    inherit: bool = True,
    report: ReportCallback | None = None,
) -> HandlerScope:
    """Aggregate results across every optimization run in the block.

    Each handler receives events from all runs in the block (sequential or
    concurrent) and is serialized across them. Blocks nest: by default a nested
    block also inherits the enclosing blocks' handlers, so they aggregate the
    nested runs too. Pass ``inherit=False`` to include only the handlers the
    nested block lists (re-list an enclosing handler to feed it explicitly).

    See [Running Optimizations](../running/running.md) for a walkthrough.

    A block claims each of its handlers for as long as it is open, and one
    `HandlerScope` object opens one block at a time. A handler that was ever
    passed to [`optimize`][ropt.simple.optimize] as a local handler cannot join
    a block afterwards; decide per handler whether it is local or shared.

    Args:
        handler:  The result handlers to share across the block, each run on the
                  block's event-loop thread.
        threaded: Handlers (one, or a sequence) to run on a worker thread instead
                  of the loop. This only helps handlers that spend real time in
                  blocking, GIL-releasing I/O (files, databases, network); for
                  in-memory work it gives no benefit under CPython's GIL. See
                  [Running Optimizations](../running/running.md#running-a-handler-in-a-thread).
        inherit:  Whether to also inherit the enclosing blocks' handlers.
        report:   An optional callback invoked with an `EvaluateResult` for each
                  function evaluation across the block's runs; return `True` from
                  it to stop the emitting run early with `USER_ABORT`.

    Returns:
        A context manager scoping the shared handlers, which binds the
        `HandlerScope` itself when used with `as`.
    """
    in_thread = (threaded,) if isinstance(threaded, EventHandler) else tuple(threaded)
    scope_handlers: list[tuple[EventHandler, bool]] = [
        *((item, False) for item in handler),
        *((item, True) for item in in_thread),
    ]
    if report is not None:
        scope_handlers.append((make_report_handler(report), False))
    return HandlerScope(scope_handlers, inherit=inherit)


def current_handlers() -> HandlerScope | None:
    """Return the innermost open `handlers` block, if any.

    Call this from a compute step or launcher to reach the shared handlers
    active in the caller's context, then wire a run with `attach_to`. Read it on
    the thread that owns the block and pass the scope to each run, so
    `optimize_many`'s worker threads forward to the same dispatcher.

    Returns:
        The innermost block's scope, or `None` when no block is open.
    """
    stack = _handler_stack.get()
    return stack[-1] if stack else None
