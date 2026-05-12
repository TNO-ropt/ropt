from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._results import Results

if TYPE_CHECKING:
    from ropt.context import EnOptContext

    from ._constraint_info import ConstraintInfo
    from ._function_evaluations import FunctionEvaluations
    from ._functions import Functions
    from ._realizations import Realizations


TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class FunctionResults(Results):
    """Store results related to function evaluations.

    The `FunctionResults` class extends the base
    [`Results`][ropt.results.Results] class to store data specific to function
    evaluations. This includes:

    1. **Evaluations:** The results of the function evaluations, including the
       variable values, objective values, and constraint values for each
       realization. See
       [`FunctionEvaluations`][ropt.results.FunctionEvaluations].

     2. **Realizations:** Information about realizations, such as weights for
         objectives and constraints and whether each realization was successful.
       See [`Realizations`][ropt.results.Realizations].

    3. **Functions:** Calculated objective and constraint function values,
       typically aggregated across realizations. See
       [`Functions`][ropt.results.Functions].

    4. **Constraint Info:** Details about constraint differences and
       violations. See [`ConstraintInfo`][ropt.results.ConstraintInfo].

    Attributes:
        evaluations:     Function-evaluation results.
        realizations:    Realization-specific parameters.
        functions:       Aggregated function values, if available.
        constraint_info: Information on constraint differences and violations.
    """

    evaluations: FunctionEvaluations
    realizations: Realizations
    functions: Functions | None
    constraint_info: ConstraintInfo | None = None

    def transform_from_optimizer(self, context: EnOptContext) -> FunctionResults:
        """Transform results from optimizer space to user space.

        This applies inverse transformations to all transformable sub-fields
        (`evaluations`, `functions`, and `constraint_info` when present).
        Realization metadata is passed through unchanged.

        Args:
            context: The context used by the source of the results.

        Returns:
            The transformed results.
        """
        evaluations = self.evaluations._transform_from_optimizer(context)  # noqa: SLF001
        functions: Functions | None = None
        if self.functions is not None:
            functions = self.functions._transform_from_optimizer(context)  # noqa: SLF001
        constraint_info: ConstraintInfo | None = None
        if self.constraint_info is not None:
            constraint_info = self.constraint_info._transform_from_optimizer(  # noqa: SLF001
                context
            )

        if evaluations is None and functions is None and constraint_info is None:
            return self

        return FunctionResults(
            batch_id=self.batch_id,
            metadata=self.metadata,
            names=self.names,
            evaluations=self.evaluations if evaluations is None else evaluations,
            realizations=self.realizations,
            functions=self.functions if functions is None else functions,
            constraint_info=(
                self.constraint_info if constraint_info is None else constraint_info
            ),
        )
