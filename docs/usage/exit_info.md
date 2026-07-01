# Exit Information

`ropt`'s optimization entry points return an
[`ExitInfo`][ropt.exit_info.ExitInfo] describing how the run ended. The
object carries two fields:

- `exit_code` — an [`ExitCode`][ropt.enums.ExitCode] indicating why the
  step terminated.
- `message` — a human-readable summary suitable for reporting to end
  users. If not supplied at construction time, `ExitInfo` fills in a
  default message derived from `exit_code`.

`ExitInfo` is a keyword-only frozen dataclass. Specialized subclasses may
be introduced for individual exit codes to carry extra diagnostic fields;
callers that only read `exit_code` and `message` continue to work
unchanged.

## Reading it

A minimal termination check:

```python
from ropt.enums import ExitCode
from ropt.workflow import BasicOptimizer

optimizer = BasicOptimizer(config, evaluator)
info = optimizer.run(initial_values)

if info.exit_code != ExitCode.OPTIMIZER_FINISHED:
    print(info.message)
```

The same object is returned by the lower-level entry points
([`OptimizationStep.run`][ropt.workflow.compute_steps.OptimizationStep.run],
[`EvaluationStep.run`][ropt.workflow.compute_steps.EvaluationStep.run],
[`EnsembleOptimizer.start`][ropt.core.EnsembleOptimizer.start]) and is
attached as the `info` attribute of the [`Abort`][ropt.exceptions.Abort]
exception when you drive those steps yourself:

```python
from ropt.exceptions import Abort

try:
    step.run(...)
except Abort as exc:
    info = exc.info
```

## Raising `Abort`

`Abort` must be initialized with an
[`ExitInfo`][ropt.exit_info.ExitInfo]. Supply an explicit `message` when
the default (derived from `exit_code`) is not descriptive enough:

```python
from ropt.enums import ExitCode
from ropt.exceptions import Abort
from ropt.exit_info import ExitInfo

# Use the default message for TOO_FEW_REALIZATIONS.
raise Abort(ExitInfo(exit_code=ExitCode.TOO_FEW_REALIZATIONS))

# Override the default with a fully custom message.
raise Abort(
    ExitInfo(
        exit_code=ExitCode.TOO_FEW_REALIZATIONS,
        message="No realizations survived the CVaR filter",
    )
)
```

## Structured info subclasses

Some exit codes are always accompanied by extra structured information —
for example, `MAX_FUNCTIONS_REACHED` carries the numeric limit that was
reached, and `ABORT_FROM_ERROR` may carry the underlying error text. For
those, `ropt` uses dedicated
[`ExitInfo`][ropt.exit_info.ExitInfo] subclasses:

| Subclass                     | Exit code                | Extra field |
| ---------------------------- | ------------------------ | ----------- |
| `MaxFunctionsReachedInfo`    | `MAX_FUNCTIONS_REACHED`  | `limit`     |
| `MaxBatchesReachedInfo`      | `MAX_BATCHES_REACHED`    | `limit`     |
| `AbortFromErrorInfo`         | `ABORT_FROM_ERROR`       | `error`     |

Each subclass:

- Fixes `exit_code` to the corresponding enum value.
- Generates a default `message` that incorporates its extra field(s).
- Accepts an explicit `message` to override that default.

Callers can dispatch on the subclass to build their own reporting:

```python
from ropt.exit_info import (
    AbortFromErrorInfo,
    ExitInfo,
    MaxBatchesReachedInfo,
    MaxFunctionsReachedInfo,
)

info: ExitInfo = optimizer.run(initial_values)

match info:
    case MaxFunctionsReachedInfo(limit=n):
        print(f"stopped after {n} function evaluations")
    case MaxBatchesReachedInfo(limit=n):
        print(f"stopped after {n} batches")
    case AbortFromErrorInfo(error=err):
        print(f"failed: {err}")
    case _:
        print(info.message)
```

The base `ExitInfo` is still valid for the exit codes that have no extra
structured information (e.g. `OPTIMIZER_FINISHED`, `USER_ABORT`).
