"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from ropt.optimization import LabelStep, Plan, PlanContext


class DefaultLabelStep(LabelStep):
    """The default label step."""

    def __init__(
        self,
        label: str,
        context: PlanContext,  # noqa: ARG002
        plan: Plan,  # noqa: ARG002
    ) -> None:
        """Initialize a default config step.

        Args:
            label:   The ID of the tracker to reset
            context: Context in which the plan runs
            plan:    The plan containing this step
        """
        self._label = label

    @property
    def label(self) -> str:
        """Get the label."""
        return self._label
