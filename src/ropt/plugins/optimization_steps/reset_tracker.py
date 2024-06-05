"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from ropt.optimization import BasicStep, Plan, PlanContext


class DefaultResetTrackerStep(BasicStep):
    """The default step for resetting trackers."""

    def __init__(
        self,
        tracker_id: str,
        context: PlanContext,  # noqa: ARG002
        plan: Plan,
    ) -> None:
        """Initialize a default config step.

        Args:
            tracker_id: The ID of the tracker to reset
            context:    Context in which the plan runs
            plan:       The plan containing this step
        """
        self._tracker_id = tracker_id
        self._plan = plan

    def run(self) -> bool:
        """Run the step.

        Returns:
            True if a user abort occurred.
        """
        self._plan.reset_tracker(self._tracker_id)
        return False
