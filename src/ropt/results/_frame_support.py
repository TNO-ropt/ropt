"""Availability of the optional frame libraries, and the message when one is missing."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Final, Literal

DataFrameEngine = Literal["pandas", "polars"]

HAVE_PANDAS: Final = find_spec("pandas") is not None
HAVE_POLARS: Final = find_spec("polars") is not None

_INSTALLED: Final[dict[str, bool]] = {"pandas": HAVE_PANDAS, "polars": HAVE_POLARS}


def engine_available(engine: DataFrameEngine) -> bool:
    """Report whether a frame library is installed.

    Args:
        engine: The frame library to check.

    Returns:
        `True` if it can be imported.
    """
    return _INSTALLED[engine]


def missing_engine_message(engine: DataFrameEngine, what: str, alternative: str) -> str:
    """Build the message for `what` needing a frame library that is not installed.

    Args:
        engine:      The frame library that `what` needs.
        what:        The function, method or class that needs it.
        alternative: How to use the other library instead, named only when that
                     library is itself installed.

    Returns:
        The message.
    """
    other = "polars" if engine == "pandas" else "pandas"
    msg = f"{what} requires the {engine} module; install ropt[{engine}]"
    return msg + (f", or {alternative}." if _INSTALLED[other] else ".")
