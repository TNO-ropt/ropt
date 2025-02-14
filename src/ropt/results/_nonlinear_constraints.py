from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


@dataclass(slots=True)
class NonlinearConstraints(ResultField):
    """This class stores non-linear constraint differences.

    Attributes:
        lower_diffs: Lower bound differences.
        upper_diffs: Upper bound differences.
    """

    lower_diffs: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,)}
    )
    upper_diffs: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,)}
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.lower_diffs = _immutable_copy(self.lower_diffs)
        self.upper_diffs = _immutable_copy(self.upper_diffs)

    @classmethod
    def create(
        cls, config: EnOptConfig, constraints: NDArray[np.float64] | None
    ) -> NonlinearConstraints | None:
        """Add constraint information.

        Args:
            config:      The ensemble optimizer configuration object.
            constraints: The constraints to check.

        Returns:
            A newly created NonlinearConstraints object.
        """
        if config.nonlinear_constraints is None or constraints is None:
            return None
        return NonlinearConstraints(
            lower_diffs=(constraints - config.nonlinear_constraints.lower_bounds),
            upper_diffs=(constraints - config.nonlinear_constraints.upper_bounds),
        )
