"""Default values used by the configuration classes.

See [Configuration Reference][config-perturbations] for these field defaults.
See [Stochastic Gradients][gradient-settings] for how they are used.

[config-perturbations]: ../optimizer_setup/configuration_sections.md#variable-perturbations
[gradient-settings]: ../optimizer_setup/gradients.md
"""

from typing import Final

from ropt.enums import BoundaryType

DEFAULT_SEED: Final = 1
"""Default seed for random number generators used by samplers."""

DEFAULT_NUMBER_OF_PERTURBATIONS: Final = 5
"""Default number of perturbations for gradient estimation."""

DEFAULT_PERTURBATION_MAGNITUDE: Final = 0.005
"""Default perturbation magnitude in the configured variable units."""

DEFAULT_PERTURBATION_BOUNDARY_TYPE: Final = BoundaryType.MIRROR
"""Default boundary handling for perturbations that violate variable bounds."""
