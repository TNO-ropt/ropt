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


@dataclass(slots=True)
class BoundConstraints(ResultField):
    """This class stores bound constraint differences.

    Attributes:
        lower_diffs: Lower bound differences.
        upper_diffs: Upper bound differences.
    """

    lower_diffs: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxis.VARIABLE,)}
    )
    upper_diffs: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxis.VARIABLE,)}
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
    ) -> BoundConstraints | None:
        """Add bound constraint information.

        Args:
            config:    The ensemble optimizer configuration object.
            variables: The variables to check.

        Returns:
            A newly created BoundConstraints object or None.
        """
        if np.all(np.isfinite(config.variables.lower_bounds)) and np.all(
            np.isfinite(config.variables.upper_bounds)
        ):
            return BoundConstraints(
                lower_diffs=variables - config.variables.lower_bounds,
                upper_diffs=variables - config.variables.upper_bounds,
            )
        return None
