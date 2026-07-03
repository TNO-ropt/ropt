"""Internal logging helpers for ropt."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    # Strip the trailing private module component so that logger names only
    # expose public package names, e.g. ropt.core._optimizer → ropt.core.
    return logging.getLogger(name.rsplit("._", 1)[0])
