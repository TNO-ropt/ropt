"""Framework and Implementations for Optimizer backends.

This module provides the necessary components for integrating optimization
algorithms into `ropt` via its plugin system. Backend plugins allow `ropt` to
utilize various optimization techniques, either built-in or provided by
third-party packages.

Backends must derive from the [`Backend`][ropt.backend._base.Backend] base class
and implement the required interface to be compatible with `ropt`. This includes
methods for initializing the backend, performing optimization steps, and
handling any specific requirements of the optimization algorithms they support.

**Built-in Backends:**

- [`SciPyBackend`][ropt.backend.scipy.SciPyBackend]: A backend that interfaces
  with optimization algorithms from the SciPy library, supporting a wide range
  of methods for unconstrained and constrained optimization.
- [`ExternalBackend`][ropt.backend.external.ExternalBackend]: A backend that
  allows optimization via an external process, enabling the use of optimization
  algorithms that may not be directly integrated into `ropt` but can be executed
  as separate processes.

**Utilities:**

The [`ropt.backend.utils`][ropt.backend.utils] module offers helper
functions for common tasks within optimizer backends, such as validating
constraint support and handling normalized constraints.
"""

from ._base import Backend

__all__ = ["Backend"]
