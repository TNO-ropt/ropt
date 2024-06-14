"""Default values used by the configuration classes."""

from typing import Final

from ropt.enums import BoundaryType, PerturbationType

DEFAULT_SEED: Final = 1
"Default random generator seed."

DEFAULT_NUMBER_OF_PERTURBATIONS: Final = 5
"""Default number of perturbations."""

DEFAULT_PERTURBATION_MAGNITUDE: Final = 0.005
"""Default perturbation magnitude."""

DEFAULT_PERTURBATION_BOUNDARY_TYPE: Final = BoundaryType.MIRROR_BOTH
"""Default perturbation boundary handling type.

See also: [`BoundaryType`][ropt.enums.BoundaryType].
"""

DEFAULT_PERTURBATION_TYPE: Final = PerturbationType.ABSOLUTE
""" Default perturbation type.

See also: [`PerturbationType`][ropt.enums.PerturbationType].
"""
