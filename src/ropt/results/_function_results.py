from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._results import Results

if TYPE_CHECKING:
    from ropt.transforms import OptModelTransforms

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

    Attributes:
        evaluations:           Results of the function evaluations.
        realizations:          The calculated parameters of the realizations.
        functions:             The calculated functions.
    """

    evaluations: FunctionEvaluations
    realizations: Realizations
    functions: Functions | None

    def transform_back(self, transforms: OptModelTransforms) -> FunctionResults:
        """Apply backward transforms to the results.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return FunctionResults(
            plan_id=self.plan_id,
            batch_id=self.batch_id,
            metadata=self.metadata,
            evaluations=self.evaluations.transform_back(transforms),
            realizations=self.realizations,
            functions=(
                None
                if self.functions is None
                else self.functions.transform_back(transforms)
            ),
        )
