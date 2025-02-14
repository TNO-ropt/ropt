from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.transforms import OptModelTransforms


@dataclass(slots=True)
class LinearConstraints(ResultField):
    """This class stores linear constraint differences.

    Attributes:
        lower_diffs: Lower bound differences.
        upper_diffs: Upper bound differences.
    """

    lower_diffs: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxis.LINEAR_CONSTRAINT,)}
    )
    upper_diffs: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxis.LINEAR_CONSTRAINT,)}
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.lower_diffs = _immutable_copy(self.lower_diffs)
        self.upper_diffs = _immutable_copy(self.upper_diffs)

    @classmethod
    def create(
        cls, config: EnOptConfig, variables: NDArray[np.float64]
    ) -> LinearConstraints | None:
        """Add linear constraint information.

        Args:
            config:    The ensemble optimizer configuration object.
            variables: The variables to check.

        Returns:
            A newly created LinearConstraints object or None.
        """
        if config.linear_constraints is None:
            return None
        values = np.matmul(config.linear_constraints.coefficients, variables)
        return LinearConstraints(
            lower_diffs=values - config.linear_constraints.lower_bounds,
            upper_diffs=values - config.linear_constraints.upper_bounds,
        )

    def transform_from_optimizer(
        self, transforms: OptModelTransforms
    ) -> LinearConstraints:
        if transforms.variables is not None:
            lower_diffs, upper_diffs = (
                transforms.variables.linear_constraints_diffs_from_optimizer(
                    self.lower_diffs, self.upper_diffs
                )
            )
            return LinearConstraints(
                lower_diffs=lower_diffs,
                upper_diffs=upper_diffs,
            )
        return self
