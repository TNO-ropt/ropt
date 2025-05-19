from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._results import Results

if TYPE_CHECKING:
    from ._constraint_info import ConstraintInfo
    from ._function_evaluations import FunctionEvaluations
    from ._functions import Functions
    from ._realizations import Realizations


TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class FunctionResults(Results):
    """Stores results related to function evaluations.

    The `FunctionResults` class extends the base
    [`Results`][ropt.results.Results] class to store data specific to function
    evaluations. This includes:

    1. **Evaluations:** The results of the function evaluations, including the
       variable values, objective values, and constraint values for each
       realization. See
       [`FunctionEvaluations`][ropt.results.FunctionEvaluations].

    2. **Realizations:** Information about the realizations, such as weights for
       objectives and constraints, and whether each realization was successful.
       See [`Realizations`][ropt.results.Realizations].

    3. **Functions:** The calculated objective and constraint function values,
       typically aggregated across realizations. See
       [`Functions`][ropt.results.Functions].

    4. **Constraint Info:** Details about constraint differences and violations.
       See [`ConstraintInfo`][ropt.results.ConstraintInfo].

    Attributes:
        evaluations:     Results of the function evaluations.
        realizations:    The calculated parameters of the realizations.
        functions:       The calculated functions.
        constraint_info: Information on constraint differences and violations.
    """

    evaluations: FunctionEvaluations
    realizations: Realizations
    functions: Functions | None
    constraint_info: ConstraintInfo | None = None

    def transform_from_optimizer(self) -> FunctionResults:
        """Apply transformations from optimizer space.

        Returns:
            The transformed results.
        """
        assert self.config.transforms is not None
        return FunctionResults(
            config=self.config,
            batch_id=self.batch_id,
            metadata=self.metadata,
            evaluations=self.evaluations.transform_from_optimizer(
                self.config.transforms
            ),
            realizations=self.realizations,
            functions=(
                None
                if self.functions is None
                else self.functions.transform_from_optimizer(self.config.transforms)
            ),
            constraint_info=(
                None
                if self.constraint_info is None
                else self.constraint_info.transform_from_optimizer(
                    self.config.transforms
                )
            ),
        )
