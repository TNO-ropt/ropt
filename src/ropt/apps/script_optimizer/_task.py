"""Definition of the ScriptTask class."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ropt.evaluator.parsl import Task


@dataclass(slots=True)
class ScriptTask(Task):
    """Task class for use by the ScriptBasedOptimizer class."""

    name: str = ""
    job_labels: tuple[str, ...] = field(default_factory=tuple)
    objective_paths: tuple[Path, ...] = field(default_factory=tuple)
    constraint_paths: tuple[Path, ...] = field(default_factory=tuple)
    logged: bool = False
    realization: int = 0
    _objectives: NDArray[np.float64] | None = None
    _constraints: NDArray[np.float64] | None = None

    def _read_file(self, path: Path) -> float:
        try:
            with path.open("r", encoding="utf-8") as file_obj:
                return float(file_obj.read())
        except BaseException as exc:  # noqa: BLE001
            self.exception = exc
            return np.nan

    def _read_results(self) -> None:
        if self.objective_paths:
            self._objectives = np.zeros(len(self.objective_paths), dtype=np.float64)
            for idx, path in enumerate(self.objective_paths):
                self._objectives[idx] = self._read_file(path)
        if self.constraint_paths:
            self._constraints = np.zeros(len(self.constraint_paths), dtype=np.float64)
            for idx, path in enumerate(self.constraint_paths):
                self._constraints[idx] = self._read_file(path)

    def get_objectives(self) -> NDArray[np.float64] | None:
        """Get the objective values.

        Returns:
            The objective values.
        """
        self._read_results()
        return self._objectives

    def get_constraints(self) -> NDArray[np.float64] | None:
        """Get the constraint values.

        Returns:
            The constraint values.
        """
        self._read_results()
        return self._constraints
