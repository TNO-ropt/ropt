"""Optimizer backends: the bridge to an external optimization library.

A backend receives an
[`OptimizationProblem`][ropt.backend.OptimizationProblem], requests function and
gradient values through the
[`OptimizerCallback`][ropt.core.OptimizerCallback] interface, and advances the
optimization from an initial variable vector. It is selected through the
`backend` field of an [`EnOptContext`][ropt.context.EnOptContext], either as an
instance or as a [`BackendConfig`][ropt.config.BackendConfig] naming a method.
The context itself never reaches the backend: `ropt` reduces it to an
`OptimizationProblem` when the run starts.

`ropt` ships [`SciPyBackend`][ropt.backend.scipy.SciPyBackend] and
[`ExternalBackend`][ropt.backend.external.ExternalBackend], which runs an
optimizer in a separate process; further backends come from plugin packages. See
[Writing a Plugin](../advanced/writing_plugins.md) for implementing one.
"""

from ._base import Backend
from ._problem import OptimizationProblem

__all__ = ["Backend", "OptimizationProblem"]
