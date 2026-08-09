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
from typing import TYPE_CHECKING

from ropt.components.event_handlers import EventDispatcher, EventForwardHandler

from ._report import make_report_handler
from ._session import _acquire_session, _release_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.components.compute_steps import ComputeStep
    from ropt.components.event_handlers import EventHandler
    from ropt.enums import EnOptEventType

    from ._report import ReportCallback
    from ._session import Session


_handler_stack: ContextVar[tuple[HandlerScope, ...]] = ContextVar(
    "ropt_simple_handlers", default=()
)


class HandlerScope:
    """A block of shared result handlers, opened by `handlers`.

    Aggregates results across every run in the block. A compute step reaches the
    active block with `current_handlers` and feeds its events to the shared
    handlers with `attach_to`.
    """

    def __init__(self, handlers: Sequence[EventHandler], *, inherit: bool) -> None:
        self._handlers = tuple(handlers)
        self._inherit = inherit
        self._session: Session | None = None
        self._session_token: Token[Session | None] | None = None
        self._stack_token: Token[tuple[HandlerScope, ...]] | None = None
        self._dispatcher: EventDispatcher | None = None
        # The handlers currently attached to this scope's dispatcher; a nested
        # block temporarily steals these (re-listed or inherited) and restores
        # them on exit.
        self.current: set[EventHandler] = set()
        self._migrated: list[tuple[EventHandler, HandlerScope]] = []

    def attach_to(self, step: ComputeStep) -> None:
        """Forward the events of a run's compute step to this block's handlers.

        A fresh forwarding handler is added per step (one handler cannot serve
        several steps), carrying only the event types the shared handlers want.

        Args:
            step: The compute step whose events feed the shared handlers.
        """
        event_types: set[EnOptEventType] = set()
        for handler in self.current:
            event_types |= handler.event_types
        if event_types:
            step.add_event_handler(
                EventForwardHandler(self.dispatcher, event_types=event_types)
            )

    @property
    def dispatcher(self) -> EventDispatcher:
        assert self._dispatcher is not None
        return self._dispatcher

    def __enter__(self) -> None:
        session, session_token = _acquire_session()
        dispatcher = EventDispatcher()
        stack = _handler_stack.get()
        migrated: list[tuple[EventHandler, HandlerScope]] = []
        try:
            added = self._attach(dispatcher, stack, migrated)
            session.open_dispatcher(dispatcher)
        except BaseException:
            self._detach(dispatcher, migrated)
            session.close_dispatcher(dispatcher)
            _release_session(session, session_token)
            raise
        self._session = session
        self._session_token = session_token
        self._dispatcher = dispatcher
        self.current = added
        self._migrated = migrated
        self._stack_token = _handler_stack.set((*stack, self))

    def _attach(
        self,
        dispatcher: EventDispatcher,
        stack: tuple[HandlerScope, ...],
        migrated: list[tuple[EventHandler, HandlerScope]],
    ) -> set[EventHandler]:
        added: set[EventHandler] = set()
        for handler in self._handlers:
            source = _find_scope(stack, handler)
            if source is not None:
                source.dispatcher.remove_event_handler(handler)
                source.current.discard(handler)
                migrated.append((handler, source))
            dispatcher.add_event_handler(handler)
            added.add(handler)
        if self._inherit:
            for source in stack:
                for handler in list(source.current):
                    source.dispatcher.remove_event_handler(handler)
                    source.current.discard(handler)
                    migrated.append((handler, source))
                    dispatcher.add_event_handler(handler)
                    added.add(handler)
        return added

    @staticmethod
    def _detach(
        dispatcher: EventDispatcher,
        migrated: list[tuple[EventHandler, HandlerScope]],
    ) -> None:
        for handler, source in migrated:
            dispatcher.remove_event_handler(handler)
            source.dispatcher.add_event_handler(handler)
            source.current.add(handler)

    def __exit__(self, *_exc: object) -> None:
        assert self._stack_token is not None
        _handler_stack.reset(self._stack_token)
        self._detach(self.dispatcher, self._migrated)
        assert self._session is not None
        self._session.close_dispatcher(self.dispatcher)
        _release_session(self._session, self._session_token)


def handlers(
    *handler: EventHandler,
    inherit: bool = True,
    report: ReportCallback | None = None,
) -> HandlerScope:
    """Aggregate results across every optimization run in the block.

    Each handler receives events from all runs in the block (sequential or
    concurrent) and is serialized across them. Blocks nest: by default a nested
    block also inherits the enclosing blocks' handlers, so they aggregate the
    nested runs too. Pass ``inherit=False`` to include only the handlers the
    nested block lists (re-list an enclosing handler to feed it explicitly).

    See [High-Level API](../usage/simple.md) for a walkthrough.

    Args:
        handler: The result handlers to share across the block.
        inherit: Whether to also inherit the enclosing blocks' handlers.
        report:  An optional callback invoked with an `EvaluateResult` for each
                 function evaluation across the block's runs.

    Returns:
        A context manager scoping the shared handlers.
    """
    scope_handlers: tuple[EventHandler, ...] = handler
    if report is not None:
        scope_handlers = (*handler, make_report_handler(report))
    return HandlerScope(scope_handlers, inherit=inherit)


def _find_scope(
    stack: tuple[HandlerScope, ...], handler: EventHandler
) -> HandlerScope | None:
    for scope in reversed(stack):
        if handler in scope.current:
            return scope
    return None


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
