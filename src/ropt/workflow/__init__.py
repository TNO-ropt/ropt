"""High-level optimization driver.

[`BasicOptimizer`][ropt.workflow.BasicOptimizer] runs a single optimization by
assembling the low-level `ropt.components` building blocks.
"""

from ._basic_optimizer import BasicOptimizer

__all__ = [
    "BasicOptimizer",
]
