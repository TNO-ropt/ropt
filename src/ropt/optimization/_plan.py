"""This module defines the Plan class."""

from __future__ import annotations

from itertools import count, dropwhile
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from numpy.random import default_rng

from ropt.config.enopt import EnOptConfig
from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.optimization import BasicStep, LabelStep, TrackerStep
from ropt.utils import update_dict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.plan import PlanConfig
    from ropt.evaluator import Evaluator
    from ropt.plugins import PluginManager
    from ropt.results import FunctionResults, Results

    from ._events import OptimizationEventBroker


class PlanContext:
    def __init__(
        self,
        events: OptimizationEventBroker,
        evaluator: Evaluator,
        seed: Optional[int],
        plugin_manager: PluginManager,
    ) -> None:
        self.events = events
        self.evaluator = evaluator
        self.rng = default_rng(DEFAULT_SEED) if seed is None else default_rng(seed)
        self.plugin_manager = plugin_manager
        self.result_id_iter = count()
        self.results: Dict[str, Optional[FunctionResults]] = {}
        self._result_key: Optional[str] = None


class Plan:
    """The plan runner."""

    def __init__(self, config: PlanConfig, context: PlanContext) -> None:
        """Initialize a plan runner.

        Args:
            config:  Optimizer configuration
            context: Context in which the plan runs
        """
        self._config = config
        self._context = context
        self._enopt_config: Optional[EnOptConfig] = None
        self._steps, self._trackers = self._create_steps(self._config.root)
        self._metadata_setters = self._get_metadata_setters()
        self._restart: bool
        self._restart_label: Optional[str]
        self._result: Optional[str] = self._trackers[-1].id if self._trackers else None

    @property
    def enopt_config(self) -> Optional[EnOptConfig]:
        """Return the optimizer configuration of this step.

        Returns:
            The optimizer configuration.
        """
        return self._enopt_config

    @property
    def results(self) -> Dict[str, Optional[FunctionResults]]:
        """Return the results.

        Returns:
            A dictionary of function results.
        """
        return self._context.results

    @property
    def final_result(self) -> Optional[FunctionResults]:
        """Return the result of the last tracker.

        Returns:
            A function result or None.
        """
        if self._result is not None:
            return self._context.results.get(self._result)
        return None

    def run(self, variables: Optional[NDArray[np.float64]] = None) -> bool:
        """Run the plan.

        Args:
            variables: Optional variables to start running with

        Returns:
            Whether a user abort occurred.
        """
        self._restart = True
        self._restart_label = None

        for tracker in self._trackers:
            tracker.reset()

        while self._restart:
            steps = self._steps
            if self._restart_label is not None:
                steps = list(
                    dropwhile(
                        lambda step: (not isinstance(step, LabelStep))
                        or (step.label != self._restart_label),
                        self._steps,
                    )
                )

            self._restart = False
            self._restart_label = None

            for step in steps:
                if isinstance(step, LabelStep):
                    continue
                if step.run() if isinstance(step, BasicStep) else step.run(variables):
                    return True
                if self._restart:
                    break

        return False

    def reset_tracker(self, tracker_id: str) -> None:
        """Reset a tracker.

        Args:
            tracker_id: The ID of the tracker to reset.
        """
        tracker = next((item for item in self._trackers if item.id == tracker_id), None)
        if tracker is not None:
            tracker.reset()

    def restart(self, label: Optional[str] = None) -> None:
        """Direct the plan to restart."""
        self._restart = True
        self._restart_label = label

    def track_results(
        self, results: Tuple[Results, ...], step_id: Optional[str]
    ) -> Tuple[Results, ...]:
        """Update the trackers.

        Args:
            results: The results to track
            step_id: ID of the step producing the results
        """
        # Run the trackers:
        if step_id is not None:
            for tracker in self._trackers:
                tracker.track_results(results, step_id)

        # Optionally let the steps set some metadata:
        for set_metadata in self._metadata_setters:
            results = set_metadata(results)

        return results

    def set_enopt_config(self, enopt_config: EnOptConfig) -> None:
        """Set the configuration.

        Args:
            enopt_config: The configuration to set
        """
        self._enopt_config = enopt_config

    def update_enopt_config(self, updates: Dict[str, Any]) -> None:
        """Update the optimizer configuration.

        Args:
            updates: Dictionary of update values
        """
        assert self._enopt_config is not None
        assert self._enopt_config.original_inputs is not None
        self._enopt_config = EnOptConfig.model_validate(
            update_dict(self._enopt_config.original_inputs, updates)
        )

    def _create_steps(
        self, plan_config: Tuple[Dict[str, Any], ...]
    ) -> Tuple[Tuple[Any, ...], Tuple[TrackerStep, ...]]:
        all_steps = [
            self._context.plugin_manager.get_plugin(
                "optimization_step", method=next(iter(step_config))
            )
            .create(self._context, self)
            .get_step(step_config)
            for step_config in plan_config
        ]
        steps: List[Any] = []
        trackers: List[TrackerStep] = []
        steps, trackers = [], []
        for step in all_steps:
            if isinstance(step, TrackerStep):
                trackers.append(step)
            else:
                steps.append(step)
        return tuple(steps), tuple(trackers)

    def _get_metadata_setters(
        self,
    ) -> Tuple[Callable[[Tuple[Results, ...]], Tuple[Results, ...]]]:
        set_metadata_attrs = tuple(
            getattr(step, "set_metadata", None) for step in self._steps
        )
        return tuple(
            set_metadata
            for set_metadata in set_metadata_attrs
            if set_metadata is not None
        )
