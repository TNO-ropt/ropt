"""Internal logging helpers for ropt."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    # Logger names expose only public-looking names: a trailing private module
    # is dropped and private packages lose their underscore, for example
    # ropt._core._optimizer becomes ropt.core.
    parts = name.split(".")
    if parts[-1].startswith("_"):
        parts.pop()
    return logging.getLogger(".".join(part.removeprefix("_") for part in parts))
