"""The optimization plan configuration class."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from pydantic import RootModel


class PlanConfig(RootModel[Tuple[Dict[str, Any], ...]]):
    """Configuration for a single optimization plan."""

    root: Tuple[Dict[str, Any], ...]
