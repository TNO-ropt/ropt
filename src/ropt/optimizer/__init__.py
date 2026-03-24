"""Framework and Implementations for Optimizer backends.

This module provides the necessary components for integrating optimization
algorithms into `ropt` via its plugin system. Optimizer plugins allow `ropt` to
utilize various optimization techniques, either built-in or provided by
third-party packages.


**Utilities:**

The [`ropt.optimizer.utils`][ropt.optimizer.utils] module offers helper
functions for common tasks within optimizer backends, such as validating
constraint support and handling normalized constraints.
"""

from ._base import Optimizer

__all__ = ["Optimizer"]
