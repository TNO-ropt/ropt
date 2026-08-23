"""Plugin support for optimizer backends.

A backend wraps one or more optimization algorithms. A
[`BackendPlugin`][ropt.plugins.backend.BackendPlugin] is a factory that creates
the [`Backend`][ropt.backend.Backend] objects doing the actual work, which the
[`PluginManager`][ropt.plugins.manager.PluginManager] discovers through the
`ropt.plugins.backend` entry point group.

`ropt` ships [`SciPyBackend`][ropt.backend.scipy.SciPyBackend], and
[`ExternalBackend`][ropt.backend.external.ExternalBackend], which runs another
backend in a separate process.

See [Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

from ._base import BackendPlugin

__all__ = [
    "BackendPlugin",
]
