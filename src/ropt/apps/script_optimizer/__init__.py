"""Simple application object to run script-based workflows with parsl."""

from ._config import ScriptOptimizerConfig
from ._script_optimizer import ScriptOptimizer

__all__ = [
    "ScriptOptimizer",
    "ScriptOptimizerConfig",
]
