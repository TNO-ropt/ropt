"""The script optimizer configuration class."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator


class ScriptOptimizerConfig(BaseModel):
    """Configuration of a ScriptOptimizer object.

    This configuration class defines a number of parameters that determine how
    the optimizer runs:

    - The working directory where the various files that the application needs
      is defined by `work_dir`.
    - The directory where the files generated by the individual evaluation jobs
      are placed is determined by the `job_dir`. If `job_dir` is `None` is set
      equal to `work_dir`. If `job_dir` is a relative path, it is set relative
      to `work_dir`.
    - Individual evaluation jobs are assigned a label, which is used to generate
      the name of the directory in `job_dir` where the files that they place are
      generated. The `job_labels` attribute is a tuple of strings that indicates
      the names of the nested directories that are generated. The string may
      contain Python replacement fields to generate specific names:
        - `batch` is replaced by the batch ID.
        - `realization` is replaced by the realization name.
        - `job` is replaced by the job ID
    - Each evaluation job receives a set of variables, written as json file with
      a name given by the `var_filename` attribute.

    Attributes:
        work_dir:     Working directory.
        job_dir:      The directory to store files generated during optimization.
        job_labels:   Label formats to use in generating filenames reports.
        var_filename: Name of the generated variable file.
    """

    work_dir: Path
    job_dir: Optional[Path] = None
    job_labels: Tuple[str, ...] = ("B{batch:04d}", "J{job:04d}")
    var_filename: Path = Path("variables")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _after_validator(self) -> ScriptOptimizerConfig:
        self.work_dir = Path(self.work_dir).resolve()
        self.job_dir = self.work_dir if self.job_dir is None else self.job_dir
        if not self.job_dir.is_absolute():
            self.job_dir = self.work_dir / self.job_dir
        return self


class ScriptEvaluatorConfig(BaseModel):
    """Configuration of the parsl evaluator used by the ScriptOptimizer.

    Attributes:
        htex_kwargs:    Keyword arguments forwarded to the htex executor.
        max_threads:    Maximum number of threads for local runs.
        worker_restart: Restart the workers every `worker_restart` batch.
        polling:        How often should be polled for status.
        max_submit:     Maximum number of variables to submit simultaneously.
    """

    htex_kwargs: Optional[Dict[str, Any]] = None
    max_threads: int = 4
    worker_restart: int = 0
    polling: float = 0.1
    max_submit: int = 500
