"""Default values used by the configuration classes.

See the [Configuration guide][config-perturbations] for detailed explanations
of these defaults and their interactions.

[config-perturbations]: ../usage/configuration.md#variable-perturbations
[config-gradient]: ../usage/configuration.md#gradient-gradientconfig
"""

from typing import Final

from ropt.enums import BoundaryType, PerturbationType

DEFAULT_SEED: Final = 1
"""Default seed for random number generators used by samplers."""

DEFAULT_NUMBER_OF_PERTURBATIONS: Final = 5
"""Default number of perturbations for gradient estimation."""

DEFAULT_PERTURBATION_MAGNITUDE: Final = 0.005
"""Default scaling factor applied to sampler-generated perturbation values."""

DEFAULT_PERTURBATION_BOUNDARY_TYPE: Final = BoundaryType.MIRROR_BOTH
"""Default boundary handling for perturbations that violate variable bounds."""

DEFAULT_PERTURBATION_TYPE: Final = PerturbationType.ABSOLUTE
"""Default interpretation of perturbation magnitudes."""
