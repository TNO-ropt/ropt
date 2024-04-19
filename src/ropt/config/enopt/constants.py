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

DEFAULT_OPTIMIZATION_BACKEND = "scipy"
"""Default optimization backend."""

DEFAULT_SAMPLER_BACKEND = "scipy"
"""Default sampler backend."""

DEFAULT_REALIZATION_FILTER_BACKEND = "default"
"""Default realization backend."""

DEFAULT_FUNCTION_TRANSFORM_BACKEND = "default"
"""Default function transform backend."""
