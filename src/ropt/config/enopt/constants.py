"""Default values used by the configuration classes."""

from ropt.enums import BoundaryType, PerturbationType

DEFAULT_SEED = 1
"Default random generator seed."

DEFAULT_NUMBER_OF_PERTURBATIONS = 5
"""Default number of perturbations."""

DEFAULT_PERTURBATION_MAGNITUDE = 0.005
"""Default perturbation magnitude."""

DEFAULT_PERTURBATION_BOUNDARY_TYPE = BoundaryType.MIRROR_BOTH
"""Default perturbation boundary handling type.

See also: [`BoundaryType`][ropt.enums.BoundaryType].
"""

DEFAULT_PERTURBATION_TYPE = PerturbationType.ABSOLUTE
""" Default perturbation type.

See also: [`PerturbationType`][ropt.enums.PerturbationType].
"""
