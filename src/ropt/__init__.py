"""The main `ropt` module, a library for ensemble based optimization.

The high-level convenience API (`ropt.optimize`, ...) lives in the
[`ropt.highlevel`][ropt.highlevel] subpackage and is lazily re-exported here, so
`import ropt` stays cheap and optional dependencies are only imported on first
use.
"""
# ruff: file-ignore[non-empty-init-module]

import logging
from typing import TYPE_CHECKING, Any

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from ropt.highlevel import EvaluateResult as EvaluateResult
    from ropt.highlevel import ObjectiveCallback as ObjectiveCallback
    from ropt.highlevel import OptimizeResult as OptimizeResult
    from ropt.highlevel import evaluate as evaluate
    from ropt.highlevel import evaluate_many as evaluate_many
    from ropt.highlevel import optimize as optimize
    from ropt.highlevel import optimize_many as optimize_many
    from ropt.highlevel import processes as processes
    from ropt.highlevel import threads as threads

_HIGHLEVEL_NAMES = (
    "EvaluateResult",
    "ObjectiveCallback",
    "OptimizeResult",
    "evaluate",
    "evaluate_many",
    "optimize",
    "optimize_many",
    "processes",
    "threads",
)

__all__ = list(_HIGHLEVEL_NAMES)


def __getattr__(name: str) -> Any:  # ruff: ignore[any-type]
    if name in _HIGHLEVEL_NAMES:
        import ropt.highlevel as _highlevel  # ruff: ignore[import-outside-top-level]

        return getattr(_highlevel, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), *_HIGHLEVEL_NAMES])
