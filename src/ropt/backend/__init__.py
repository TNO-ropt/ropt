"""Public API for optimizer backend implementations.

Backends define how `ropt` runs an optimization algorithm against the problem
described by an [`EnOptContext`][ropt.context.EnOptContext] object. A backend
manages the optimizer lifecycle, requests function and gradient evaluations
through the core callback interface, and advances the optimization from an
initial variable vector toward a solution.

**Core Interface**

All backend implementations inherit from the
[`Backend`][ropt.backend.Backend] base class, which defines the backend
lifecycle (`__init__`, `init`, `start`), validation hook (`validate_options`),
and capability flags (`allow_nan`, `is_parallel`).

**Integration with Optimization**

Backends are accessed via an
[`EnOptContext`][ropt.context.EnOptContext] object through its `backend`
field. A backend is instantiated either directly as an object or via a
[`BackendConfig`][ropt.config.BackendConfig] object, which is used by the
plugin system to create an instance based on the configured backend method
string.

During execution, a backend uses the
[`OptimizerCallback`][ropt.core.OptimizerCallback] interface to request
objective, constraint, and gradient evaluations from the `ropt` core.

**Built-in and Custom Backends**

`ropt` includes two built-in backends:

- [`SciPyBackend`][ropt.backend.scipy.SciPyBackend]: Uses optimization methods
  provided by SciPy.
- [`ExternalBackend`][ropt.backend.external.ExternalBackend]: Delegates the
  optimization loop to an external executable or process.

Users can implement custom backends by subclassing `Backend`. Those subclasses
can be instantiated directly and passed into an
[`EnOptContext`][ropt.context.EnOptContext] object through its `backend`
field. Registering a custom backend with the plugin system is optional and
only required when the backend should be selected and configured via
`BackendConfig` objects instead of being instantiated explicitly by the user.
"""

from ._base import Backend

__all__ = ["Backend"]
