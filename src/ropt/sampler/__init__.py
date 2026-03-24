"""Provides functionality for sampler objects.

Samplers are used by the optimization process to generate perturbed variable
vectors. This module allows for the extension of `ropt` with custom samplers.
"""

from ._base import Sampler

__all__ = [
    "Sampler",
]
