"""Framework and Implementations for Optimizer backends.

This module provides the necessary components for integrating optimization
algorithms into `ropt` via its plugin system. Backend plugins allow `ropt` to
utilize various optimization techniques, either built-in or provided by
third-party packages.


**Utilities:**

The [`ropt.backend.utils`][ropt.backend.utils] module offers helper
functions for common tasks within optimizer backends, such as validating
constraint support and handling normalized constraints.
"""

from ._base import Backend

__all__ = ["Backend"]
