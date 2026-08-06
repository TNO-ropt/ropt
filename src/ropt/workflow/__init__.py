"""High-level optimization driver and utilities.

[`BasicOptimizer`][ropt.workflow.BasicOptimizer] runs a single optimization by
assembling the low-level `ropt.components` building blocks; the plugin-finder
helpers resolve and validate optimizer and sampler methods, and
[`dispatch_tasks`][ropt.workflow.dispatch_tasks] runs ad-hoc parallel work on an
executor.
"""

from ._basic_optimizer import BasicOptimizer
from ._dispatch_tasks import dispatch_tasks
from ._utils import (
    find_backend_plugin,
    find_sampler_plugin,
    validate_backend_options,
)

__all__ = [
    "BasicOptimizer",
    "dispatch_tasks",
    "find_backend_plugin",
    "find_sampler_plugin",
    "validate_backend_options",
]
