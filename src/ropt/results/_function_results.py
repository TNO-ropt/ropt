from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._results import Results

if TYPE_CHECKING:
    from ropt.transforms import OptModelTransforms

    from ._constraint_info import ConstraintInfo
    from ._function_evaluations import FunctionEvaluations
    from ._functions import Functions
    from ._realizations import Realizations


TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class FunctionResults(Results):
    """The `FunctionResults` class stores function related results.

    This class contains  the following additional information:

    1. The results of the function evaluations.
    2. The parameters of the realizations, such as weights for objectives and
       constraints, and realization failures.
    3. The calculated objective and constraint function values.
    4. Information on the differences and violations of any constraints.

    Attributes:
        evaluations:     Results of the function evaluations.
        realizations:    The calculated parameters of the realizations.
        functions:       The calculated functions.
        constraint_info: Bound constraint info.
    """

    evaluations: FunctionEvaluations
    realizations: Realizations
    functions: Functions | None
    constraint_info: ConstraintInfo | None = None

    def transform_from_optimizer(
        self, transforms: OptModelTransforms
    ) -> FunctionResults:
        """Apply transformations from optimizer space.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return FunctionResults(
            eval_id=self.eval_id,
            batch_id=self.batch_id,
            metadata=self.metadata,
            evaluations=self.evaluations.transform_from_optimizer(transforms),
            realizations=self.realizations,
            functions=(
                None
                if self.functions is None
                else self.functions.transform_from_optimizer(transforms)
            ),
            constraint_info=(
                None
                if self.constraint_info is None
                else self.constraint_info.transform_from_optimizer(transforms)
            ),
        )
