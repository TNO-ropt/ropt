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
    """Results of a function evaluation batch.

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    Attributes:
        evaluations:     Per-realization evaluation data.
        realizations:    Realization activity and weights.
        functions:       Aggregated function values, or `None` if all failed.
        constraint_info: Constraint differences and violations, if applicable.
    """

    evaluations: FunctionEvaluations
    realizations: Realizations
    functions: Functions | None
    constraint_info: ConstraintInfo | None = None

    def unscale(self, context: EnOptContext) -> FunctionResults:
        """Unscale the results.

        Unscales the sub-fields that carry values (`evaluations`, `functions`,
        and `constraint_info` when present). Realization metadata is passed
        through unchanged.

        Args:
            context: The context used by the source of the results.

        Returns:
            The unscaled results.
        """
        evaluations = self.evaluations._unscale(context)  # ruff: ignore[private-member-access]
        functions: Functions | None = None
        if self.functions is not None:
            functions = self.functions._unscale(context)  # ruff: ignore[private-member-access]
        constraint_info: ConstraintInfo | None = None
        if self.constraint_info is not None:
            constraint_info = self.constraint_info._unscale(  # ruff: ignore[private-member-access]
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
